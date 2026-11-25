#!/usr/bin/env python
# Copyright (c) 2022-2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.


"""
Optimized IRIS/Sigmet Data I/O
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Same data model as the ``iris`` engine, with a faster reader:

- the ray offsets of a sweep are found with one zero-copy 16-bit view of the
  file instead of a record slice per compression code;
- in-record data runs are copied from that view, not through a memmap slice;
- coordinates requested repeatedly only decode the sweep's first moment once;
- all sweeps of a volume share one open file, so the file's headers are parsed
  once per volume instead of once per sweep.

Select it with ``engine="iris_opt"``. The ``iris`` engine is unchanged.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

__all__ = [
    "IrisOptBackendEntrypoint",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import struct
import warnings
from collections import OrderedDict

import numpy as np
from xarray.backends.common import datatree_from_dict_with_io_cleanup
from xarray.backends.file_manager import CachingFileManager
from xarray.backends.store import StoreBackendEntrypoint

from .common import (
    LOCK_PARAM_DOC,
    REINDEX_PARAMS_DOC,
    SITE_COORDS_PARAM_DOC,
    _apply_reindex_coord,
    _apply_site_as_coords,
    _build_groups_dict,
    _compose_docstring,
    _get_reindex_coord,
    _open_sweeps,
    _resolve_sweeps,
)
from .iris import (
    BIN2,
    EXTENDED_HEADER_V0,
    EXTENDED_HEADER_V1,
    EXTENDED_HEADER_V2,
    IRIS_LOCK,
    LEN_RAW_PROD_BHDR,
    RECORD_BYTES,
    IrisBackendEntrypoint,
    IrisRawFile,
    IrisStore,
    _azimuth_centers,
    _get_fmt_string,
    _get_iris_group_names,
    _get_struct_dtype,
    _IRIS_UNITS,
)


class IrisOptRawFile(IrisRawFile):
    """IRIS RAW file served from one 16-bit view of the file."""

    def __init__(self, filename, **kwargs):
        self._fh16 = None
        super().__init__(filename, **kwargs)

    def get_compression_code(self):
        """Read and return data compression code.

        Returns
        -------
        cmp_msb : bool
            True, if MSB is set.
        cmp_val : int
            Value of data compression code.
        """
        if self._rh.pos >= RECORD_BYTES:
            # the code is the first word of the next record, after its raw
            # product header
            self.init_next_record()
            self.raw_product_bhdrs.append(self.get_raw_prod_bhdr())
        cmp_val = int(self.fh16[(self._rh.recnum * RECORD_BYTES + self._rh.pos) // 2])
        self._rh.pos += 2
        cmp_msb = cmp_val < 0
        if cmp_msb:
            cmp_val += 32767 + 1
        if self._debug:
            print(
                "--- Sub CMP Code:",
                cmp_msb,
                cmp_val,
                self._rh.recpos - 1,
                self._rh.recpos,
            )
        return cmp_msb, cmp_val

    def get_ray(self, data, skip=False):
        """Retrieve single ray.

        Returns
        -------
        data : array-like
            Numpy array containing data of one ray.
        """
        ray_pos = 0
        recnum = self.rh.recnum
        recoff = self.rh.pos

        cmp_msb, cmp_val = self.get_compression_code()

        # ray is missing
        if (cmp_val == 1) & (cmp_msb == 0):
            if self._debug:
                print("ray missing")
            return recnum, recoff

        while not ((cmp_val == 1) & (not cmp_msb)):
            if cmp_msb:
                if self._debug:
                    print(
                        f"--- Add {cmp_val} WORDS at range {ray_pos}, "
                        f"record {self._rh.recpos}:{self._rh.recpos + cmp_val}:"
                    )
                if skip:
                    self.skip_from_record(cmp_val, "int16")
                else:
                    # a run inside one record is a view of the file words; the
                    # record/header handling of read_from_record is only needed
                    # when the run crosses a record boundary
                    start = self.rh.recnum * RECORD_BYTES + self.rh.pos
                    if (
                        self.rh.pos + 2 * cmp_val <= RECORD_BYTES
                        and start // 2 + cmp_val <= self.fh16.size
                    ):
                        data[ray_pos : ray_pos + cmp_val] = self.fh16[
                            start // 2 : start // 2 + cmp_val
                        ]
                        self.rh.pos += 2 * cmp_val
                    else:
                        data[ray_pos : ray_pos + cmp_val] = self.read_from_record(
                            cmp_val, "int16"
                        )
            # # compressed zeros follow
            # # can be skipped, if data array is created all zeros
            # else:
            #     if not skip:
            #         if self._debug:
            #             print(
            #                 "--- Add {0} Zeros at range {1}, record {2}:{3}:"
            #                 "".format(
            #                     cmp_val, ray_pos, self._rh.recpos, self._rh.recpos + 1
            #                 )
            #             )
            #         if cmp_val + ray_pos > self.nbins + 6:
            #             return recnum, recoff
            #         data[ray_pos : ray_pos + cmp_val] = 0

            ray_pos += cmp_val

            # read next compression code
            cmp_msb, cmp_val = self.get_compression_code()

        return recnum, recoff

    @property
    def fh16(self):
        """The file as little-endian 16-bit words, without a copy.

        The ray streams are 16-bit words; a view of the whole file reads a
        compression code with one array index instead of a record slice and
        dtype view per word, which made scanning the offsets dominant.
        """
        if self._fh16 is None:
            self._fh16 = np.frombuffer(self._fh, dtype="<i2")
        return self._fh16

    def _get_ray_record_offsets(self, sweep_number):
        """Get record numbers and ray offsets for a specific sweep.

        The ray streams of the data types interleave, each one a run-length
        coded sequence ending with the code 1 (4.4.4, page 66). The length of
        a ray is stored nowhere, so every code of the sweep is walked once to
        find the start of the next ray.
        """
        sweep = self.data[sweep_number]

        # get rays per available data type
        rays_per_data_type = [
            d["number_rays_file_expected"] for d in sweep["ingest_data_hdrs"].values()
        ]
        ray_count = sum(rays_per_data_type)
        rnums = np.empty((ray_count,), dtype=int)
        roffs = np.empty((ray_count,), dtype=int)
        words = self.fh16
        rec = sweep["record_number"]
        pos = sweep["first_ray_byte_offset"]

        # get all ray record numbers and offsets
        for i in range(ray_count):
            rnums[i] = rec
            roffs[i] = pos
            while True:
                # a code never straddles a record: the next record starts
                # with its raw product header
                while pos >= RECORD_BYTES:
                    pos = LEN_RAW_PROD_BHDR + pos - RECORD_BYTES
                    rec += 1
                offset = rec * RECORD_BYTES + pos
                if offset >= 2 * words.size:
                    raise EOFError(f"Unexpected file end detected at record {rec}")
                code = int(words[offset // 2])
                pos += 2
                if code < 0:
                    # code + 32768 words of data follow
                    pos += 2 * (code + 32768)
                elif code == 1:
                    break
            # a ray with no data is a single code 1

        ray_offsets = np.stack([rnums, roffs], axis=-1)
        ray_offsets = ray_offsets.reshape(
            rays_per_data_type[0], len(rays_per_data_type), 2
        )
        ray_offsets = ray_offsets.swapaxes(0, 1)
        offsets = OrderedDict()
        for i, data_type in enumerate(sweep["ingest_data_hdrs"].keys()):
            offsets[data_type] = ray_offsets[i]
        sweep["ray_offsets"] = offsets

    def get_moment(self, sweep_number, moment):
        """Load a single moment of a given sweep.

        Parameters
        ----------
        sweep_number : int
            Sweep Number.
        moment : str
            Data Type to retrieve.

        """
        sweep = self.data[sweep_number]

        # create new sweep_data OrderedDict
        if not sweep.get("sweep_data", False):
            sweep["sweep_data"] = OrderedDict()

        # check if moment exist and return early
        data = sweep["sweep_data"].get(moment, False)
        if data is not False:
            return

        all_types = list(sweep["ingest_data_hdrs"].keys())

        # if coordinate is wanted, get from first moment
        if moment in ["azimuth", "elevation", "dtime", "dtime_ms", "range"]:
            moment = all_types[0]

        # the coordinates come with the first moment of the sweep, so a repeat
        # request for them must not decode its rays again
        if moment in sweep["sweep_data"] and "azimuth" in sweep["sweep_data"]:
            return

        idx = self.data_types.index(moment)
        prod = self.data_types_dict[idx]
        moment_num = all_types.index(moment)
        moment_type = self.data_types_dict[moment_num]

        phdr = self._product_hdr

        rays = sweep["ingest_data_hdrs"][moment]["number_rays_file_expected"]
        bins = phdr["product_end"]["number_bins"]

        raw_data = np.zeros((rays, bins + 6), dtype="int16")

        # get raw_data
        if not sweep.get("ray_offsets", False):
            self._get_ray_record_offsets(sweep_number)
        for i, (rnum, roff) in enumerate(sweep["ray_offsets"][moment_type["name"]]):
            if self.rh.recnum != rnum:
                self.init_record(rnum)
            self.rh.pos = roff
            self.get_ray(raw_data[i])

        xhdr_type = phdr["product_configuration"]["product_specific_info"]["xhdr_type"]

        data = raw_data[:, 6:]

        # extended header contains exact ray times with ms resolution
        if moment == "DB_XHDR":
            if xhdr_type == 0:
                ext_hdr = EXTENDED_HEADER_V0
            elif xhdr_type == 1:
                ext_hdr = EXTENDED_HEADER_V1
            elif xhdr_type == 2:
                ext_hdr = EXTENDED_HEADER_V2
                warnings.warn(
                    "xradar: Sigmet/Iris Extended Header V2 not implemented "
                    "completely. The customer specified reminder will not be "
                    "decoded. Please use `rawdata=True` to load undecoded data."
                )
            else:
                raise ValueError(f"wradlib: unknown extended header type V{xhdr_type}")
            len_ext_hdr = struct.calcsize(_get_fmt_string(ext_hdr))
            dtype = _get_struct_dtype(ext_hdr)
            if not self.rawdata:
                data = data[:, : len_ext_hdr // 2].copy().view(dtype)

        # remove missing rays marked as 0 in the following arrays
        idx = np.where(raw_data[:, 4] != 0)[0]
        if idx.size:
            data = data[idx]
            raw_data = raw_data[idx]

        sweep_data = sweep["sweep_data"]

        sweep_data[moment] = self.decode_data(data, prod)

        # get millisecond times if available
        if moment == "DB_XHDR":
            sweep_data["dtime_ms"] = np.squeeze(data.T["dtime_ms"])

        if sweep_data.get("azimuth", False) is False:
            dec_data = self.decode_data(raw_data[:, :4], BIN2)
            azi_start = dec_data[:, 0]
            azi_stop = dec_data[:, 2]
            # correct elevation for negative angles
            # see https://github.com/wradlib/wradlib/issues/560
            ele_start = dec_data[:, 1]
            ele_start[ele_start > 180] -= 360
            ele_stop = dec_data[:, 3]
            ele_stop[ele_stop > 180] -= 360

            if self.rawdata:
                # raw BIN2 counts, not degrees: keep the previous arithmetic
                # (counter-clockwise rays are not handled)
                zero_index = np.where(azi_start - azi_stop > 5)
                azi_stop2 = azi_stop.copy()
                azi_stop2[zero_index[0]] += 360
                az = (azi_start + azi_stop2) / 2.0
                az[az >= 360] -= 360
            else:
                az = _azimuth_centers(azi_start, azi_stop)
            el = (ele_start + ele_stop) / 2.0

            sweep_data["azi_start"] = azi_start
            sweep_data["azi_stop"] = azi_stop
            sweep_data["azimuth"] = az
            sweep_data["ele_start"] = ele_start
            sweep_data["ele_stop"] = ele_stop
            sweep_data["elevation"] = el
            sweep_data["rbins"] = raw_data[:, 4]
            sweep_data["dtime"] = raw_data[:, 5]


class _SharedManager:
    """Reference-counted handle on a file manager shared by several stores.

    ``IrisOptBackendEntrypoint.open_groups_as_dict`` reads a file's headers
    once and serves every sweep from the same :class:`IrisOptRawFile`, instead
    of parsing the whole file once per sweep. Closing one sweep dataset must
    not close the file the other sweeps are still reading from, so the manager
    is only closed once the last store releases it.
    """

    def __init__(self, manager):
        self._manager = manager
        self._refs = 1

    def retain(self):
        self._refs += 1
        return self

    def acquire_context(self, needs_lock=True):
        return self._manager.acquire_context(needs_lock)

    def acquire(self, needs_lock=True):
        return self._manager.acquire(needs_lock)

    def close(self, needs_lock=True):
        self._refs -= 1
        if self._refs <= 0:
            self._manager.close(needs_lock)


class IrisOptStore(IrisStore):
    """Store for reading IRIS sweeps through :class:`IrisOptRawFile`."""

    def __init__(self, manager, group=None, lock=IRIS_LOCK):
        super().__init__(manager, group=group, lock=lock)
        self._closed = False

    @classmethod
    def open(cls, filename, mode="r", group=None, lock=None, **kwargs):
        if lock is None:
            lock = IRIS_LOCK
        manager = CachingFileManager(IrisOptRawFile, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group, lock=lock)

    def close(self, **kwargs):
        # a dataset may be closed twice (e.g. explicitly and with its tree);
        # releasing a shared manager twice would close the other sweeps' file
        if self._closed:
            return
        self._closed = True
        self._manager.close(**kwargs)


class IrisOptBackendEntrypoint(IrisBackendEntrypoint):
    """Xarray BackendEntrypoint for IRIS/Sigmet data, optimized reader."""

    description = "Open IRIS/Sigmet files in Xarray, optimized reader"

    def guess_can_open(self, filename_or_obj):
        """The optimized reader is opt-in and is never auto-discovered.

        ``iris`` claims IRIS files for engine discovery; selecting this reader
        is a deliberate choice (``engine="iris_opt"``), so it must not make
        discovery ambiguous.
        """
        return False

    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        group=None,
        lock=None,
        first_dim="auto",
        reindex_coord=None,
        reindex_angle=False,
        fix_second_angle=False,
        site_as_coords=True,
        optional=True,
        _manager=None,
    ):
        if _manager is None:
            store = IrisOptStore.open(
                filename_or_obj,
                group=group,
                lock=lock,
                loaddata=False,
            )
        else:
            store = IrisOptStore(_manager.retain(), group=group, lock=lock)

        store_entrypoint = StoreBackendEntrypoint()

        ds = store_entrypoint.open_dataset(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

        # reassign azimuth/elevation/time coordinates
        ds = ds.assign_coords({"azimuth": ds.azimuth})
        ds = ds.assign_coords({"elevation": ds.elevation})
        ds = ds.assign_coords({"time": ds.time})

        # drop DB_XHDR for now
        ds = ds.drop_vars("DB_XHDR", errors="ignore")

        ds.encoding["engine"] = "iris_opt"

        # handle duplicates and reindex
        reindex_coord = _get_reindex_coord(reindex_coord, reindex_angle)
        if decode_coords and reindex_coord:
            ds = _apply_reindex_coord(ds, reindex_coord)

        ds.attrs.pop("elevation_lower_limit", None)
        ds.attrs.pop("elevation_upper_limit", None)

        # handling first dimension
        dim0 = "elevation" if ds.sweep_mode.load() == "rhi" else "azimuth"

        # todo: could be optimized
        if first_dim == "time":
            ds = ds.swap_dims({dim0: "time"})
            ds = ds.sortby("time")
        else:
            ds = ds.sortby(dim0)

        # assign geo-coords
        ds = _apply_site_as_coords(ds, site_as_coords)
        ds.attrs.update(store.get_calibration_parameters())
        ds.attrs.update(store.get_radar_parameters())

        # ensure close works
        ds._close = store.close

        return ds

    def open_groups_as_dict(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        lock=None,
        first_dim="auto",
        reindex_coord=None,
        reindex_angle=False,
        fix_second_angle=False,
        site_coords=True,
        sweep=None,
        optional=True,
        optional_groups=False,
    ):
        sweeps = _resolve_sweeps(sweep, lambda: _get_iris_group_names(filename_or_obj))

        ds_kwargs = dict(
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
            lock=lock,
            first_dim=first_dim,
            reindex_coord=_get_reindex_coord(reindex_coord, reindex_angle),
            fix_second_angle=fix_second_angle,
            site_as_coords=site_coords,
        )

        # parse the file (all product/ingest headers) once for all sweeps
        manager = _SharedManager(
            CachingFileManager(
                IrisOptRawFile, filename_or_obj, mode="r", kwargs=dict(loaddata=False)
            )
        )
        try:
            ls_ds = _open_sweeps(
                lambda swp: self.open_dataset(
                    filename_or_obj, group=swp, _manager=manager, **ds_kwargs
                ),
                sweeps,
            )
        finally:
            manager.close()
        return _build_groups_dict(
            ls_ds, optional=optional, optional_groups=optional_groups, units=_IRIS_UNITS
        )

    def open_datatree(self, filename_or_obj, **kwargs):
        groups_dict = self.open_groups_as_dict(filename_or_obj, **kwargs)
        return datatree_from_dict_with_io_cleanup(groups_dict)


IrisOptBackendEntrypoint.open_groups_as_dict.__doc__ = _compose_docstring(
    "Open an Iris/Sigmet RAW file as a CfRadial2-shaped dict of group datasets, "
    "using the optimized reader.",
    REINDEX_PARAMS_DOC,
    SITE_COORDS_PARAM_DOC,
    LOCK_PARAM_DOC,
)
IrisOptBackendEntrypoint.open_datatree.__doc__ = (
    "Open an Iris/Sigmet RAW file as :py:class:`xarray.DataTree`. "
    "See :meth:`open_groups_as_dict` for keyword arguments.\n"
)