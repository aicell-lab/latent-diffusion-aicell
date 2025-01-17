#!/usr/bin/env python3
import os
import pandas as pd


def main():
    data_dir = "local_cpg_data"  # directory containing subfolders for each plate
    chem_annot_csv = "chemical_annotations.csv"
    barcode_platemap_csv = "barcode_platemap.csv"  # this maps plate_id -> platemap_name
    out_csv = "pairs_allplates.csv"

    # 1) Read chemical annotations
    df_chem = pd.read_csv(chem_annot_csv)

    # 2) Read the barcode_platemap.csv that links each plate_id to a platemap file
    #    Suppose columns: [Plate_Map_Name, Assay_Plate_Barcode, etc.]
    df_barcode = pd.read_csv(barcode_platemap_csv)
    # We might need to create a "plate_id" without commas
    df_barcode["plate_id"] = (
        df_barcode["Assay_Plate_Barcode"].astype(str).str.replace(",", "")
    )
    # Example: row["Plate_Map_Name"] = "H-BIOA-004-3"

    # 3) Collect all "pairs" data in a list to concatenate later
    all_dfs = []

    # 4) Loop over every file in data_dir that looks like "load_data_*.csv"
    for fname in os.listdir(data_dir):
        if not fname.startswith("load_data_") or not fname.endswith(".csv"):
            continue

        load_data_csv = os.path.join(data_dir, fname)
        # e.g. "load_data_24277.csv" => plate_id = "24277"
        plate_id = fname.replace("load_data_", "").replace(".csv", "")
        print(f"Processing plate_id={plate_id}, file={load_data_csv}")

        # 4A) Identify the matching platemap
        row = df_barcode.loc[df_barcode["plate_id"] == plate_id]
        if row.empty:
            print(
                f"Warning: No row in barcode_platemap.csv for plate {plate_id}. Skipping."
            )
            continue

        platemap_name = row["Plate_Map_Name"].iloc[0]
        platemap_file = os.path.join(data_dir, f"{platemap_name}.txt")

        if not os.path.exists(platemap_file):
            print(f"Warning: Platemap file {platemap_file} not found. Skipping.")
            continue

        # 4B) Read the load_data.csv, skipping the top row
        df_load = pd.read_csv(load_data_csv, header=None, sep=",", skiprows=1).rename(
            columns={
                0: "channelDNA_filename",
                2: "channelER_filename",
                4: "channelRNA_filename",
                6: "channelAGP_filename",
                8: "channelMito_filename",
                10: "plate_id",
                11: "well",
                12: "site",
            }
        )

        # 4C) Build absolute file paths
        def build_abs_path(filename, plate):
            return os.path.join(data_dir, str(plate), filename)

        df_load["channelDNA"] = df_load.apply(
            lambda row: build_abs_path(row["channelDNA_filename"], row["plate_id"]),
            axis=1,
        )
        df_load["channelER"] = df_load.apply(
            lambda row: build_abs_path(row["channelER_filename"], row["plate_id"]),
            axis=1,
        )
        df_load["channelRNA"] = df_load.apply(
            lambda row: build_abs_path(row["channelRNA_filename"], row["plate_id"]),
            axis=1,
        )
        df_load["channelAGP"] = df_load.apply(
            lambda row: build_abs_path(row["channelAGP_filename"], row["plate_id"]),
            axis=1,
        )
        df_load["channelMito"] = df_load.apply(
            lambda row: build_abs_path(row["channelMito_filename"], row["plate_id"]),
            axis=1,
        )

        # 4D) Read platemap (tab-delimited)
        df_map = pd.read_csv(platemap_file, sep="\t")
        df_load["well"] = df_load["well"].str.upper()

        df_merged = pd.merge(
            df_load, df_map, left_on="well", right_on="well_position", how="left"
        ).rename(columns={"broad_sample": "BROAD_ID"})

        # 4E) Merge with chemical annotations
        df_final = pd.merge(df_merged, df_chem, on="BROAD_ID", how="left")

        # 4F) Pick final columns; also keep plate_id in the final table
        df_out = df_final[
            [
                "plate_id",
                "well",
                "site",
                "channelDNA",
                "channelER",
                "channelRNA",
                "channelAGP",
                "channelMito",
                "BROAD_ID",
                "CPD_NAME",
                "CPD_NAME_TYPE",
                "CPD_SMILES",
            ]
        ]
        all_dfs.append(df_out)

    # 5) Concatenate all plates into one DataFrame
    if len(all_dfs) == 0:
        print("No plates processed. Exiting.")
        return

    df_conc = pd.concat(all_dfs, ignore_index=True)
    print(f"Merged total rows: {len(df_conc)}")

    # 6) Save to CSV
    df_conc.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df_conc)} rows.")


if __name__ == "__main__":
    main()
