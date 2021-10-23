#!/usr/bin/env python

import sys
import os
import glob

import argparse
from multiprocessing import Pool
import pandas as pd
import numpy as np
import pyarrow
import pyarrow.parquet

def recast_uint(df):
    for column, dtype in zip(df.columns, df.dtypes):
        if(dtype == np.uint8):
            df[column] = df[column].astype(np.int8)
        if(dtype == np.uint16):
            df[column] = df[column].astype(np.int16)
        elif(dtype == np.uint32):
            df[column] = df[column].astype(np.int32)
        elif(dtype == np.uint64):
            df[column] = df[column].astype(np.int64)

def convert_bulk_parquet(bulk_parquet_filename, pos_parquet_filename,
                      data_parquet_filename):

    # TODO: pos_parquet_filename is vestigal, delete it

    df = pyarrow.parquet.read_table(bulk_parquet_filename).to_pandas()

    if len(df) == 0:
        print(f'No data in {bulk_parquet_filename}')
        return

#    if data_parquet_filename is None:
#        column_list = ['matchid', 'ra', 'dec']
#    else:
#        column_list = None

    columns_to_rename = ["hmjd", "mag", "magerr", "clrcoeff", "catflags"] # vector columns
    filter_map = {1: "g", 2: "r", 3: "i"}
    filters_in_datafile = list(set(df['filterid']))
    assert(len(filters_in_datafile) == 1)
    filter_number = filters_in_datafile[0]
    filter_string = filter_map[filter_number]

    rcids = list(set(df['rcid']))
    assert(len(rcids) == 1)
    rcid = np.int16(rcids[0])
    fieldids = list(set(df['fieldid']))
    assert(len(fieldids) == 1)
    fieldid = np.int16(fieldids[0])


    # blow up scalar columns to vector columns
    df[f'rcid_{filter_string}'] = df.apply(lambda x: [rcid for i in range(len(x['mag']))], axis=1)
    df[f'fieldid_{filter_string}'] = df.apply(lambda x: [fieldid for i in range(len(x['mag']))], axis=1)

    # catflags are a list of uints--change to ints
    df[f'catflags'] = df[f'catflags'].apply(lambda x: list(map(np.int16, x)))


    df.rename(columns={column: f"{column}_{filter_string}" for column 
        in columns_to_rename }, inplace=True)

    for n in set((1,2,3)) - set((filter_number,)):
        set_filter_string = filter_map[n]
        for column in columns_to_rename:
            datatype = df[f"{column}_{filter_string}"].dtype
            df[f"{column}_{set_filter_string}"] = df['objra'].apply(lambda x: [])

        df[f"rcid_{set_filter_string}"] = df['objra'].apply(lambda x: [])
        df[f"fieldid_{set_filter_string}"] = df['objra'].apply(lambda x: [])

    df.drop(columns=['filterid','fieldid','rcid','nepochs'],inplace=True)

    # no longer needed, I think, since all the arrays are object...
    recast_uint(df)



    if not os.path.exists(os.path.dirname(data_parquet_filename)):
        os.makedirs(os.path.dirname(data_parquet_filename))

    schema = pyarrow.Schema.from_pandas(df)
    table = pyarrow.Table.from_pandas(df, schema)
    pyarrow.parquet.write_table(table, data_parquet_filename, flavor='spark')

if __name__ == '__main__':

    default_input_basepath = ("/data/epyc/data/ztf_matchfiles/"
                          "msip_dr7/irsa.ipac.caltech.edu")
    default_output_basepath = "/data/epyc/data/ztf_matchfiles/scratch_dr7_bulk_ztf_parquet_for_axs"
    default_glob_pattern = "[01]/field*/*.parquet"

    parser = argparse.ArgumentParser(description="Convert bulk ZTF public data release parquet into parquet needed for AXS",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--no-data", dest="no_data", action="store_true",
                        help="Suppress saving the output photometry files, only store positions")
    parser.add_argument("--glob", dest="glob_pattern", action="store",
                        help="Glob pattern for searching the input directory",
                        type=str, default=default_glob_pattern)
    parser.add_argument("--input-path", dest="input_basepath", action="store",
                        help="Input directory",
                        type=str, default=default_input_basepath)
    parser.add_argument("--output-path", dest="output_basepath", action="store",
                        help="Output directory",
                        type=str, default=default_output_basepath)
    parser.add_argument("--nprocs", type=int, default=1,
                        help="Number of parallel processes to use")
    args = parser.parse_args()

    input_files = glob.iglob(os.path.join(args.input_basepath, args.glob_pattern))

    # This is not great coding style...
    def process_wrapper(bulk_parquet_path):
        output_file_pytable = bulk_parquet_path.replace(os.path.normpath(args.input_basepath),
                                                     os.path.normpath(args.output_basepath))
        output_pos_filename = output_file_pytable.replace(".parquet", "_pos.parquet")

        if args.no_data:
            output_data_filename = None
            if(os.path.exists(output_pos_filename)):
                return
        else:
            output_data_filename = output_file_pytable.replace(".parquet", "_data.parquet")
            if(os.path.exists(output_data_filename)):
                return

#        print(bulk_parquet_path, output_data_filename)
        convert_bulk_parquet(bulk_parquet_path, output_pos_filename,
                          output_data_filename)

    if args.nprocs > 1:
        with Pool(args.nprocs) as p:
            p.map(process_wrapper, input_files)
    else:
        for filename in input_files:
            process_wrapper(filename)

