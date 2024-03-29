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

    columns_to_rename = ["mjd", "mag", "magerr", "clrcoeff", "catflags"] # vector columns
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

    df.rename(columns={'hmjd': 'mjd', 'objra': 'ra', 'objdec': 'dec'},
        inplace=True)

    df.rename(columns={column: f"{column}_{filter_string}" for column 
        in columns_to_rename }, inplace=True)

    for n in set((1,2,3)) - set((filter_number,)):
        set_filter_string = filter_map[n]
        for column in columns_to_rename:
            datatype = df[f"{column}_{filter_string}"].dtype
            df[f"{column}_{set_filter_string}"] = df['ra'].apply(lambda x: [])

        df[f"rcid_{set_filter_string}"] = df['ra'].apply(lambda x: [])
        df[f"fieldid_{set_filter_string}"] = df['ra'].apply(lambda x: [])

    df.drop(columns=['filterid','fieldid','rcid','nepochs'],inplace=True)

    # no longer needed, I think, since all the arrays are object...
    recast_uint(df)



    if not os.path.exists(os.path.dirname(data_parquet_filename)):
        try:
            os.makedirs(os.path.dirname(data_parquet_filename))
        except FileExistsError:
            pass

    schema = pyarrow.Schema.from_pandas(df)

    # unfortunately the empty filter columns above come through to pyarrow as 
    # type list<item: null>, which fails when joining against parquet files
    # from other filters downstream.  So coerce the schemas to be correct

    for n in set((1,2,3)) - set((filter_number,)):
        set_filter_string = filter_map[n]
        for column in columns_to_rename:
            if column == 'catflags':
                dtype = pyarrow.int16()
            elif column == 'mjd':
                dtype = pyarrow.float64()
            else:
                dtype = pyarrow.float32()
            field_name = f"{column}_{set_filter_string}"
            field_index = schema.get_field_index(field_name)
            schema = schema.set(field_index, pyarrow.field(field_name,
                pyarrow.list_(dtype)))

        for column in ['rcid', 'fieldid']:
                field_name = f"{column}_{set_filter_string}"
                field_index = schema.get_field_index(field_name)
                schema = schema.set(field_index, pyarrow.field(field_name,
                        pyarrow.list_(pyarrow.int16())))

    table = pyarrow.Table.from_pandas(df, schema)
    pyarrow.parquet.write_table(table, data_parquet_filename, flavor='spark')

if __name__ == '__main__':

    dr = 'dr14'

    default_input_basepath = ("/data/epyc/data/ztf_matchfiles/"
                          f"msip_{dr}/irsa.ipac.caltech.edu")
    default_output_basepath = f"/data/epyc/data/ztf_matchfiles/scratch_{dr}_bulk_ztf_parquet_for_axs"
    default_glob_pattern = "[01]/field*/*.parquet"

    parser = argparse.ArgumentParser(description="Convert bulk ZTF public data release parquet into parquet needed for AXS",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--no-data", dest="no_data", action="store_true",
                        help="Suppress saving the output photometry files, only store positions")
    parser.add_argument("--check-output", dest="check_output", action="store_true",
                        help="Check saved parquet files for errors")
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
        output_data_filename = output_file_pytable.replace(".parquet", "_data.parquet")

        if args.check_output:
            if(os.path.exists(output_data_filename)):
                try: 
                    df = pyarrow.parquet.read_table(output_data_filename)
                    del df
                except:
                    print('Failed to read file: ',output_data_filename)
                    os.remove(output_data_filename)
                    convert_bulk_parquet(bulk_parquet_path, output_pos_filename,
                          output_data_filename)

            else:
                print('Missing output file: ',output_data_filename)
                #convert_bulk_parquet(bulk_parquet_path, output_pos_filename,
                #          output_data_filename)
            return

        if args.no_data:
            output_data_filename = None
            if(os.path.exists(output_pos_filename)):
                return
        else:
            if(os.path.exists(output_data_filename)):
                return

        convert_bulk_parquet(bulk_parquet_path, output_pos_filename,
                          output_data_filename)

    if args.nprocs > 1:
        with Pool(args.nprocs) as p:
            p.map(process_wrapper, input_files)
    else:
        for filename in input_files:
            process_wrapper(filename)

