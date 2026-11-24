#!/usr/bin/env python
# Copyright (c) 2022-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
IRIS/Sigmet optimized reader (engine ``iris_mod``)
==================================================

Drop-in replacement of the ``iris`` engine that returns identical data but opens
files much faster. It is opt-in: ``xd.open_datatree(file, engine="iris_mod")``.
The ``iris`` engine, and automatic engine discovery, are unchanged.

Why the ``iris`` engine is slow
-------------------------------
Profiling a 16 MB volume showed that most of the time is not disk I/O but the
pure-Python scan that finds where each ray starts: one ``get_ray`` call per ray,
and per ray several 2-byte compression words, each read through the call chain
``get_compression_code -> read_from_record -> array_from_record ->
IrisRecord.read -> numpy.memmap.__getitem__``. This costs a few microseconds of
interpreter and memmap-subclass overhead per 2-byte word, ~400 000 times.
A second scan, at file open, unpacks the block header of every record through
the generic ``_unpack_dictionary``.

What this module changes
------------------------
* ``IrisRawFileMod.get_ray`` walks the compression words with
  ``struct.Struct.unpack_from`` directly on the whole-file buffer, tracking
  ``(record number, byte position)`` in local variables, and copies data runs with
  one ``numpy.frombuffer`` per run. The reader state (``rh.recnum``/``rh.pos``) is
  written back once per ray, so callers see exactly the state the original leaves.
* ``IrisRawFileMod.get_data_headers`` unpacks the record block headers with one
  precompiled ``struct.Struct`` and only builds an ``IrisRecord`` for records
  that start a new sweep.

Everything else (header parsing, decoding, xarray store, entrypoint) is inherited.

Known differences from ``iris``
-------------------------------
* ``raw_product_bhdrs`` gets one entry per record from ``get_data_headers`` (same as
  ``iris``), but ``iris`` additionally appends duplicate entries each time ``get_ray``
  crosses a record boundary. ``iris_mod`` does not. Nothing in xradar reads them there.
* A truncated file raises ``EOFError`` as before; the message may name another record.
"""

__all__ = [
    "IrisModBackendEntrypoint",
]

import struct
from collections import OrderedDict

import numpy as np

from .iris import (
    LEN_RAW_PROD_BHDR,
    RAW_PROD_BHDR,
    RECORD_BYTES,
    IrisBackendEntrypoint,
    IrisRawFile,
    IrisStore,
    _get_fmt_string,
)

# One signed/unsigned 16 bit little-endian word. In a ray, the most significant bit
# of a compression word flags "N data words follow", otherwise "N zero words".
_WORD           = struct.Struct("<H")
_RAW_PROD_BHDR  = struct.Struct(_get_fmt_string(RAW_PROD_BHDR))
_DATA_FLAG      = 0x8000
_COUNT_MASK     = 0x7FFF
_END_OF_RAY     = 1  # a "1 zero word" code ends the ray (or marks a missing ray)
_BYTES_PER_WORD = 2


class IrisRawFileMod(IrisRawFile):
    """IrisRawFile with a fast ray scan and a fast record header scan."""

    def _raw_prod_bhdr_keys(self):
        """Keys of the unpacked block header, as ``_unpack_dictionary`` returns them."""
        if self._rawdata:
            return list(RAW_PROD_BHDR)
        return [
            key
            for key in RAW_PROD_BHDR
            if not key.startswith(("spare", "reserved"))
        ]

    def get_data_headers(self):
        """Load all sweep's `ingest_data_header` from file.

        Same result as :meth:`IrisRawFile.get_data_headers`, without creating an
        ``IrisRecord`` and a generic unpack for every one of the file's records.
        """
        try:
            sweeps = self.loaddata.get("sweep", None)
        except AttributeError:
            sweeps = None

        buffer        = self.fh
        file_bytes    = buffer.shape[0]
        keys          = self._raw_prod_bhdr_keys()
        keep          = [i for i, key in enumerate(RAW_PROD_BHDR) if key in keys]
        record_count  = int(np.ceil(self.structure_size / RECORD_BYTES))
        current_sweep = 0

        # records 0 and 1 are the product and ingest headers, rays start at record 2
        for record_number in range(2, record_count):
            offset = record_number * RECORD_BYTES
            if offset + RECORD_BYTES > file_bytes:
                raise EOFError(
                    f"Unexpected file end detected at record {record_number}"
                )
            values = _RAW_PROD_BHDR.unpack_from(buffer, offset)
            raw_prod_bhdr = OrderedDict((keys[j], values[i]) for j, i in enumerate(keep))
            self.raw_product_bhdrs.append(raw_prod_bhdr)

            sweep_number = raw_prod_bhdr["sweep_number"]
            if sweep_number == current_sweep:
                continue
            current_sweep = sweep_number
            if sweeps is not None and current_sweep not in sweeps:
                continue
            # a new sweep: its ingest data headers follow the record block header
            self.init_record(record_number)
            self.rh.pos = LEN_RAW_PROD_BHDR
            sweep = OrderedDict()
            sweep["ingest_data_hdrs"]      = self.get_ingest_data_headers()
            sweep["record_number"]         = self.rh.recnum
            sweep["first_ray_byte_offset"] = self.rh.pos
            self._data[current_sweep] = sweep

        self.init_record(1)

    def get_ray(self, data, skip=False):
        """Retrieve single ray.

        Parameters
        ----------
        data : array-like
            int16 row to fill with the ray words (ignored if ``skip``).
        skip : bool
            Only move the reader to the end of the ray, do not copy data.

        Returns
        -------
        recnum, recoff : int
            Record number and byte offset where the ray starts.
        """
        buffer        = self.fh
        record_number = self.rh.recnum
        position      = self.rh.pos
        start         = (record_number, position)
        ray_position  = 0

        try:
            while True:
                # next compression word; a word at the record end is the first of
                # the next record, after its block header
                if position >= RECORD_BYTES:
                    record_number += 1
                    position       = LEN_RAW_PROD_BHDR
                (code,)   = _WORD.unpack_from(buffer, record_number * RECORD_BYTES + position)
                position += _BYTES_PER_WORD

                if code & _DATA_FLAG:
                    remaining = code & _COUNT_MASK
                    # the data words may continue in the following records
                    while remaining > 0:
                        if position >= RECORD_BYTES:
                            record_number += 1
                            position       = LEN_RAW_PROD_BHDR
                        words = min(remaining, (RECORD_BYTES - position) // _BYTES_PER_WORD)
                        if not skip:
                            data[ray_position : ray_position + words] = np.frombuffer(
                                buffer,
                                dtype  = "<i2",
                                count  = words,
                                offset = record_number * RECORD_BYTES + position,
                            )
                        ray_position += words
                        position     += words * _BYTES_PER_WORD
                        remaining    -= words
                else:
                    # `code` zero words follow; data is initialized with zeros
                    if code == _END_OF_RAY:
                        break
                    ray_position += code
        except (struct.error, ValueError) as error:
            raise EOFError(
                f"Unexpected file end detected at record {record_number}"
            ) from error

        # leave the reader where the original implementation leaves it
        if record_number != self.rh.recnum:
            self.init_record(record_number)
        self.rh.pos = position
        return start


class IrisStoreMod(IrisStore):
    """IrisStore reading with :class:`IrisRawFileMod`."""

    raw_file_class = IrisRawFileMod


class IrisModBackendEntrypoint(IrisBackendEntrypoint):
    """Xarray BackendEntrypoint for IRIS/Sigmet data, optimized reader.

    Same data as the ``iris`` engine. Select it explicitly with
    ``engine="iris_mod"``; it is never picked by automatic engine discovery.
    """

    description = "Open IRIS/Sigmet files in Xarray (optimized reader)"
    store_class = IrisStoreMod
    engine_name = "iris_mod"

    def guess_can_open(self, filename_or_obj):
        """Never claim a file, the ``iris`` engine does that."""
        return False
