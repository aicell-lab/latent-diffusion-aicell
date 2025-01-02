import pandas as pd
from pathlib import Path
import re

# source: https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods
channel_map_cpg0000 = {
    4: {
        "DNA": 5,  # ch05 => DNA
        "ER": 4,  # ch04 => ER
        "RNA": 3,  # ch03 => RNA
        "AGP": 2,  # ch02 => AGP
        "Mito": 1,  # ch01 => Mito
        "BF1": 6,  # ch06 => Brightfield plane 1
        "BF2": 7,  # ch07 => Brightfield plane 2
        "BF3": 8,  # ch08 => Brightfield plane 3
    }
}


def unify_channels_vectorized_cpg0000(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized approach to assign columns (Image_Mito, Image_AGP, etc.)
    from "Image_chNN" based on channel_map_cpg0000.

    We assume df["source_number"] is a float/int that matches the keys in channel_map_cpg0000.
    """
    all_orgs = set()
    for s_num, mapping in channel_map_cpg0000.items():
        all_orgs.update(mapping.keys())

    for org in all_orgs:
        df[f"Image_{org}"] = None

    # For each source_number in the map, do a vectorized assignment
    for s_num, mapping in channel_map_cpg0000.items():
        mask = df["source_number"] == s_num
        for org, ch_num in mapping.items():
            src_col = f"Image_ch0{ch_num}" if ch_num < 10 else f"Image_ch{ch_num}"
            dest_col = f"Image_{org}"
            if src_col in df.columns:
                df.loc[mask, dest_col] = df.loc[mask, src_col]

    return df


def load_image_paths(parquet_file: str) -> pd.DataFrame:
    """Load the image paths DataFrame."""
    print(f"Loading image paths from {parquet_file}...")
    image_df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(image_df):,} image records.")
    return image_df


def extract_plate_barcode(plate_id: str) -> str:
    """Extract the plate barcode from the Metadata_Plate field."""
    # Pattern: BR00116991A__2020-11-05T19_51_35-Measurement1
    match = re.match(r"(BR\d+\w?)__", plate_id)
    if match:
        return match.group(1)
    else:
        return plate_id  # Return the original if no match


def load_batch_metadata(metadata_dir: str) -> pd.DataFrame:
    """Load and process batch metadata."""
    metadata_dir = Path(metadata_dir)
    platemaps_dir = metadata_dir / "platemaps"
    batches = [d for d in platemaps_dir.iterdir() if d.is_dir()]
    print(f"Found {len(batches)} batches in {platemaps_dir}.")

    all_plate_metadata = []

    for batch in batches:
        batch_name = batch.name
        print(f"\nProcessing batch: {batch_name}")
        barcode_path = batch / "barcode_platemap.csv"
        if not barcode_path.exists():
            print(f"  Warning: No barcode_platemap.csv found in {batch_name}")
            continue
        barcode_df = pd.read_csv(barcode_path)
        print(f"  Loaded barcode_platemap.csv with {len(barcode_df)} rows.")

        platemap_dir = batch / "platemap"
        if not platemap_dir.exists():
            print(f"  Warning: No platemap directory found in {batch_name}")
            continue

        for idx, row in barcode_df.iterrows():
            plate_barcode = row["Assay_Plate_Barcode"]
            platemap_name = row["Plate_Map_Name"]
            platemap_file = platemap_dir / f"{platemap_name}.txt"
            if not platemap_file.exists():
                print(f"    Warning: Platemap file {platemap_file} does not exist.")
                continue
            platemap_df = pd.read_csv(platemap_file, sep="\t")
            # Ensure well_position is uppercase (e.g., 'A01')
            platemap_df["well_position"] = platemap_df["well_position"].str.upper()
            print(f"    Loaded {platemap_name} with {len(platemap_df)} rows.")
            platemap_df["Metadata_Plate"] = plate_barcode
            platemap_df["Metadata_Batch"] = batch_name
            platemap_df["Plate_Map_Name"] = platemap_name
            # Infer PlateType from Plate_Map_Name
            if "compound" in platemap_name.lower():
                platemap_df["Metadata_PlateType"] = "COMPOUND"
            elif "orf" in platemap_name.lower():
                platemap_df["Metadata_PlateType"] = "ORF"
            elif "crispr" in platemap_name.lower():
                platemap_df["Metadata_PlateType"] = "CRISPR"
            else:
                platemap_df["Metadata_PlateType"] = "UNKNOWN"
            all_plate_metadata.append(platemap_df)

    if all_plate_metadata:
        plate_metadata_df = pd.concat(all_plate_metadata, ignore_index=True)
        print(f"\nCombined plate metadata has {len(plate_metadata_df):,} rows.")
    else:
        plate_metadata_df = pd.DataFrame()
        print("\nNo plate metadata found.")

    return plate_metadata_df


def load_external_metadata(metadata_dir: str) -> dict:
    """Load external metadata for compounds, ORFs, and CRISPR."""
    metadata_dir = Path(metadata_dir) / "external_metadata"
    metadata = {}

    compound_meta_file = metadata_dir / "JUMP-Target-1_compound_metadata.tsv"
    compound_targets_file = metadata_dir / "JUMP-Target-1_compound_metadata_targets.tsv"
    compound_df = pd.read_csv(compound_meta_file, sep="\t")
    compound_targets_df = pd.read_csv(compound_targets_file, sep="\t")

    compound_full_df = compound_df.merge(
        compound_targets_df,
        on=[
            "broad_sample",
            "InChIKey",
            "pert_iname",
            "pubchem_cid",
            "pert_type",
            "control_type",
            "smiles",
        ],
        how="outer",
        suffixes=("", "_y"),
    )
    # Drop duplicate columns if any
    compound_full_df = compound_full_df.loc[
        :, ~compound_full_df.columns.str.endswith("_y")
    ]
    metadata["compound"] = compound_full_df
    print(f"Loaded compound metadata with {len(compound_full_df):,} rows.")

    orf_meta_file = metadata_dir / "JUMP-Target-1_orf_metadata.tsv"
    orf_df = pd.read_csv(orf_meta_file, sep="\t")
    metadata["orf"] = orf_df
    print(f"Loaded ORF metadata with {len(orf_df):,} rows.")

    crispr_meta_file = metadata_dir / "JUMP-Target-1_crispr_metadata.tsv"
    crispr_df = pd.read_csv(crispr_meta_file, sep="\t")
    metadata["crispr"] = crispr_df
    print(f"Loaded CRISPR metadata with {len(crispr_df):,} rows.")

    return metadata


def merge_perturbation_metadata(
    df: pd.DataFrame, external_metadata: dict
) -> pd.DataFrame:
    """Merge the DataFrame with external perturbation metadata."""
    for perturbation_type in ["compound", "orf", "crispr"]:
        mask = df["Plate_Map_Name"].str.contains(
            perturbation_type, case=False, na=False
        )
        perturbation_df = df[mask].copy()
        if not perturbation_df.empty:
            print(
                f"Merging {len(perturbation_df):,} rows with {perturbation_type} metadata..."
            )
            external_df = external_metadata[perturbation_type]

            external_df_renamed = external_df.rename(
                columns=lambda x: f"{x}_external" if x != "broad_sample" else x
            )
            merged_df = perturbation_df.merge(
                external_df_renamed,
                on="broad_sample",
                how="left",
                validate="many_to_one",
            )

            columns_to_update = [
                col for col in external_df.columns if col != "broad_sample"
            ]
            for col in columns_to_update:
                df_col = f"Metadata_{col}"
                ext_col = f"{col}_external"
                if ext_col in merged_df.columns:
                    df.loc[mask, df_col] = merged_df[ext_col].values

    return df


def finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up the DataFrame and prepare for saving."""
    # Drop unnecessary columns
    columns_to_drop = ["well_position", "Metadata_Plate_plate"]
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    df["Metadata_Plate"] = df["Assay_Plate_Barcode"]

    df = df.rename(
        columns={
            "broad_sample": "Metadata_broad_sample",
            "solvent": "Metadata_solvent",
            "Metadata_smiles": "Metadata_SMILES",
        }
    )

    metadata_columns = [col for col in df.columns if col.startswith("Metadata_")]
    image_columns = [col for col in df.columns if col.startswith("Image_")]
    other_columns = [
        col for col in df.columns if col not in metadata_columns + image_columns
    ]

    df = df[metadata_columns + other_columns + image_columns]

    return df


SCRIPT_DIR = Path(__file__).parent
metadata_dir = SCRIPT_DIR / "metadata/cpg0000-jump-pilot"
output_file = SCRIPT_DIR / "data/cpg0000_dataframe.parquet"
input_file = SCRIPT_DIR / "data/cpg0000-jump-pilot-source_4.parquet"


def main():
    # parquet_file = "data/cpg0000-jump-pilot-source_4.parquet"
    # output_file = "data/cpg0000_dataframe.parquet"

    image_df = load_image_paths(input_file)

    print("\nExtracting plate barcodes from image DataFrame...")
    image_df["Assay_Plate_Barcode"] = image_df["Metadata_Plate"].apply(
        extract_plate_barcode
    )

    image_df["Metadata_Well"] = image_df["Metadata_Well"].str.upper()

    plate_metadata_df = load_batch_metadata(metadata_dir)

    if plate_metadata_df.empty:
        print("No plate metadata available. Exiting.")
        return

    print("\nMerging image paths with plate metadata...")
    merged_df = image_df.merge(
        plate_metadata_df,
        left_on=["Assay_Plate_Barcode", "Metadata_Well"],
        right_on=["Metadata_Plate", "well_position"],
        how="left",
        suffixes=("", "_plate"),
    )

    external_metadata = load_external_metadata(metadata_dir)

    big_df = merge_perturbation_metadata(merged_df, external_metadata)

    big_df = finalize_dataframe(big_df)

    big_df["Metadata_Source"] = "source_4"  # Hard-coded for cpg0000
    big_df["source_number"] = 4  # Hard-coded for cpg0000
    big_df = unify_channels_vectorized_cpg0000(big_df)
    col_to_drop = [c for c in big_df.columns if c.startswith("Image_ch0")]
    big_df.drop(columns=col_to_drop, inplace=True, errors="ignore")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final DataFrame to {output_file}...")
    big_df.to_parquet(output_file)
    print("Done!")

    print(f"\nFinal DataFrame has {len(big_df):,} rows.")
    print("Columns:")
    print(big_df.columns.tolist())


if __name__ == "__main__":
    main()
