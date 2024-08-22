#!/bin/bash
set -euov pipefail
proj=$(cat "tmp/projection.txt")

script_dir=$(realpath $(dirname "$0"))
echo $script_dir   

$script_dir/submodules/seismic-waveform-factory/scripts/compute_multi_cmt.py spatial extracted_output/dyn-kinmod_extracted-fault.xdmf 1 tmp/depth_vs_rigidity.txt --DH 10 --proj "${proj}" --NZ 4

# Use globbing directly within the loop to capture filenames
for filename in extracted_output/dyn*-fault.xdmf; do
    echo "Processing file: $filename"
    $script_dir/submodules/seismic-waveform-factory/scripts/compute_multi_cmt.py spatial "$filename" 1 tmp/depth_vs_rigidity.txt --DH 10 --proj "${proj}" --NZ 4
done
mv PointSource* tmp
$script_dir/dynworkflow/add_source_files_to_waveform_config.py
export OMP_NUM_THREADS=$(grep -c ^processor /proc/cpuinfo)
$script_dir/submodules/seismic-waveform-factory/scripts/generate_figure_synthetics.py waveforms_config_sources.ini
$script_dir/dynworkflow/compile_scenario_macro_properties.py extracted_output

# Read each line from selected_output.txt and process it
#while IFS= read -r faultoutput; do
#    echo "$faultoutput"
#done < tmp/selected_output.txt

#$seissolpath/preprocessing/science/automated_workflow_dyn_from_kin/generate_teleseismic_config_from_usgs.py
#$teleseismicspath/select_stations_azimuthal.py teleseismic_config.ini 6
#$teleseismicspath/generate_figure_teleseismic_synthetics.py teleseismic_config.ini
