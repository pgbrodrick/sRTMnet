using Sobol
using DelimitedFiles
using JSON
using ArgParse2

function index(arr, val)
    return findall(arr .== val)[1]
end

parser = ArgumentParser(prog = "Sobol sequence template generator", description = "Generate .json files for a sobol sequence")
add_argument!(parser, "--output_base_dir", type=String, default = "modtran_sobol_lut")
add_argument!(parser, "--config_name", type=String, default = "data/complete_modtran_template.json")
add_argument!(parser, "--breakout_chunk_size", type=Int64, default = -1)
add_argument!(parser, "--total_output_samples", type=Int64, default = 20)
args = parse_args(parser)

base_config = JSON.parsefile(args.config_name)

num_samples_out = args.total_output_samples
num_samples_sobol = num_samples_out * 10

# Define sequence
to_sensor_azimuth = 180
sequence_names = ["solar_zenith", "to_sensor_zenith", "solar_azimuth", "altitude", "elevation", "water_vapor", "AOD"]
sequence_upper_bounds = [cosd(0), cosd(0), cosd(0), log(99.9), log(8.85), 4.5, 1.]
sequence_lower_bounds = [cosd(80), cosd(50), cosd(359), log(0.01), log(0.001), 0.1, 0.01]
ss = SobolSeq(sequence_lower_bounds, sequence_upper_bounds)

# Populate sequence
base_sobol_sequence = zeros(num_samples_sobol, length(sequence_lower_bounds))
for i ∈ 1:size(base_sobol_sequence)[1]
    next!(ss, @view base_sobol_sequence[i,:])
end

# Trim out non-physical components (e.g., elevation > observation altitude)
final_sobol_seq = @views base_sobol_sequence[base_sobol_sequence[:,index(sequence_names, "elevation")] .> base_sobol_sequence[:,index(sequence_names, "altitude")], :][1:num_samples_out, :]
final_sobol_seq[:,1:3] = acosd.(final_sobol_seq[:,1:3])
final_sobol_seq[:,4:5] = exp.(final_sobol_seq[:,4:5])


for i in 1:size(final_sobol_seq)[1]
    output_config = copy(base_config)

    aod = round(final_sobol_seq[i,index(sequence_names, "AOD")], digits=4)
    wv = round(final_sobol_seq[i,index(sequence_names, "water_vapor")], digits=4)
    to_sensor_zenith = round(final_sobol_seq[i,index(sequence_names, "to_sensor_zenith")], digits=4)
    to_solar_azimuth = round(final_sobol_seq[i,index(sequence_names, "solar_azimuth")], digits=4)
    to_solar_zenith = round(final_sobol_seq[i,index(sequence_names, "solar_zenith")], digits=4)
    altitude = round(final_sobol_seq[i,index(sequence_names, "altitude")], digits=4)
    elevation = round(final_sobol_seq[i,index(sequence_names, "elevation")], digits=4)

    output_config["MODTRAN"][1]["MODTRANINPUT"]["AEROSOLS"]["VIS"] = -1 * aod
    output_config["MODTRAN"][1]["MODTRANINPUT"]["ATMOSPHERE"]["H2OSTR"] = wv
    output_config["MODTRAN"][1]["MODTRANINPUT"]["GEOMETRY"]["OBSZEN"] = to_sensor_zenith
    output_config["MODTRAN"][1]["MODTRANINPUT"]["GEOMETRY"]["TRUEAZ"] = to_sensor_azimuth 
    output_config["MODTRAN"][1]["MODTRANINPUT"]["GEOMETRY"]["PARM1"] = output_config["MODTRAN"][1]["MODTRANINPUT"]["GEOMETRY"]["TRUEAZ"] - to_solar_azimuth  + 180
    output_config["MODTRAN"][1]["MODTRANINPUT"]["GEOMETRY"]["PARM2"] = to_solar_zenith
                            
    output_config["MODTRAN"][1]["MODTRANINPUT"]["GEOMETRY"]["H1ALT"] = altitude
    output_config["MODTRAN"][1]["MODTRANINPUT"]["SURFACE"]["GNDALT"] = elevation

    name = "AERFRAC_2-$(aod)_GNDALT-$(elevation)_H1ALT-$(altitude)_H2OSTR-$(wv)_senzen-$(to_sensor_zenith)_solzen-$(to_solar_zenith)_solzen-$(to_solar_azimuth)_senzen-$(to_sensor_azimuth)"
    output_config["MODTRAN"][1]["MODTRANINPUT"]["NAME"] = name

    open(joinpath(args.output_base_dir, "$(name).json"), "w") do io
        JSON.print(io, output_config, 2)
    end;

end

