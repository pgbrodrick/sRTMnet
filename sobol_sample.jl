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
add_argument!(parser, "--total_output_samples", type=Int64, default = 20)
add_argument!(parser, "--breakout_chunk_size", type=Int64, default = 20)
add_argument!(parser, "--sixs", type=Int64, default = 0)
add_argument!(parser, "--log", type=Int64, default = 0)
args = parse_args(parser)

base_config = JSON.parsefile(args.config_name)

num_samples_out = args.total_output_samples
num_samples_sobol = num_samples_out * 10

# Define sequence
to_sensor_azimuth = 180
sequence_names = ["solar_zenith", "to_sensor_zenith", "solar_azimuth", "altitude", "elevation", "water_vapor", "AOD"]

#if args.log == 1
#    sequence_upper_bounds = [cosd(0), cosd(130), cosd(0), log(99.9), log(8.85), 4.5, 1.]
#    sequence_lower_bounds = [cosd(80), cosd(180), cosd(180), log(0.01), log(0.001), 0.1, 0.01]
#else
#    sequence_upper_bounds = [cosd(0), cosd(130), cosd(0), 99.9, 8.85, 4.5, 1.]
#    sequence_lower_bounds = [cosd(80), cosd(180), cosd(180), 0.01, 0.001, 0.1, 0.01]
#end

if args.log == 1
    sequence_upper_bounds = [cosd(0), cosd(130), cosd(0), log(99.9), log(8.85), 4.5, 1.]
    sequence_lower_bounds = [cosd(80), cosd(180), cosd(180), log(0.01), log(0.001), 0.1, 0.01]
else
    sequence_upper_bounds = [cosd(0), cosd(130), cosd(0), 99.9, 8.85, 7.0, 1.5]
    sequence_lower_bounds = [cosd(80), cosd(180), cosd(180), 0.01, 0.001, 0.1, 0.01]
end

ss = SobolSeq(sequence_lower_bounds, sequence_upper_bounds)

# Populate sequence
base_sobol_sequence = zeros(num_samples_sobol, length(sequence_lower_bounds))
for i in 1:size(base_sobol_sequence)[1]
    next!(ss, @view base_sobol_sequence[i,:])
end

# Trim out non-physical components (e.g., elevation > observation altitude)
final_sobol_seq = @views base_sobol_sequence[base_sobol_sequence[:,index(sequence_names, "elevation")] .< base_sobol_sequence[:,index(sequence_names, "altitude")], :][1:num_samples_out, :]
final_sobol_seq[:,1:3] = acosd.(final_sobol_seq[:,1:3])

if args.log == 1
    final_sobol_seq[:,4:5] = exp.(final_sobol_seq[:,4:5])
end

for i in 1:size(final_sobol_seq)[1]
    output_config = copy(base_config)

    aod = round(final_sobol_seq[i,index(sequence_names, "AOD")], digits=4)
    wv = round(final_sobol_seq[i,index(sequence_names, "water_vapor")], digits=4)
    to_sensor_zenith = round(final_sobol_seq[i,index(sequence_names, "to_sensor_zenith")], digits=4)
    to_solar_azimuth = round(final_sobol_seq[i,index(sequence_names, "solar_azimuth")], digits=4)
    to_solar_zenith = round(final_sobol_seq[i,index(sequence_names, "solar_zenith")], digits=4)
    altitude = round(final_sobol_seq[i,index(sequence_names, "altitude")], digits=4)
    elevation = round(final_sobol_seq[i,index(sequence_names, "elevation")], digits=4)
    name = "IND_$(i)_AERFRAC_2-$(aod)_GNDALT-$(elevation)_H1ALT-$(altitude)_H2OSTR-$(wv)_senzen-$(to_sensor_zenith)_solzen-$(to_solar_zenith)_solzen-$(to_solar_azimuth)_senzen-$(to_sensor_azimuth)"

    if args.sixs == 0
        output_config["MODTRAN"][1]["MODTRANINPUT"]["AEROSOLS"]["VIS"] = -1 * aod
        output_config["MODTRAN"][1]["MODTRANINPUT"]["ATMOSPHERE"]["H2OSTR"] = wv
        output_config["MODTRAN"][1]["MODTRANINPUT"]["GEOMETRY"]["OBSZEN"] = to_sensor_zenith
        output_config["MODTRAN"][1]["MODTRANINPUT"]["GEOMETRY"]["TRUEAZ"] = to_sensor_azimuth 
        output_config["MODTRAN"][1]["MODTRANINPUT"]["GEOMETRY"]["PARM1"] = output_config["MODTRAN"][1]["MODTRANINPUT"]["GEOMETRY"]["TRUEAZ"] - to_solar_azimuth  + 180
        output_config["MODTRAN"][1]["MODTRANINPUT"]["GEOMETRY"]["PARM2"] = to_solar_zenith
                                
        output_config["MODTRAN"][1]["MODTRANINPUT"]["GEOMETRY"]["H1ALT"] = altitude
        output_config["MODTRAN"][1]["MODTRANINPUT"]["SURFACE"]["GNDALT"] = elevation

        output_config["MODTRAN"][1]["MODTRANINPUT"]["NAME"] = name

        if args.breakout_chunk_size == -1
            open(joinpath(args.output_base_dir, "$(name).json"), "w") do io
                JSON.print(io, output_config, 2)
            end;
        else
            outbase = "$(args.output_base_dir)_$(Int64(floor((i-1) / args.breakout_chunk_size)))"

            if isdir(outbase) == false
                mkdir(outbase)
            end

            open(joinpath(outbase, "$(name).json"), "w") do io
                JSON.print(io, output_config, 2)
            end;
        end
    else
        outstr="0 (User defined)
$to_solar_zenith $to_solar_azimuth $(180-to_sensor_zenith) $to_sensor_azimuth 6 1
8  (User defined H2O, O3)
$wv, 0.30
1
0
$aod
$elevation (target level)
-$altitude (sensor level)
-$wv, -0.30
$aod
-2 
0.35
2.5
0 Homogeneous surface
0 (no directional effects)
0
0
0
-1 No atm. corrections selected"
        if args.breakout_chunk_size == -1
            input_name = joinpath(args.output_base_dir, "$(name).inp")
            output_name = joinpath(args.output_base_dir, "$(name)")
            sh_name = joinpath(args.output_base_dir, "$(name).sh")
            open(input_name, "w") do io
               write(io, outstr)
            end;
            cmdstr="/beegfs/store/shared/sixs/sixsV2.1 < $input_name > $output_name"
            open(sh_name, "w") do io
               write(io, cmdstr)
            end;
        else
            outbase = "$(args.output_base_dir)_$(Int64(floor((i-1) / args.breakout_chunk_size)))"

            input_name = joinpath(outbase, "$(name).inp")
            output_name = joinpath(outbase, "$(name)")
            sh_name = joinpath(outbase, "$(name).sh")

            if isdir(outbase) == false
                mkdir(outbase)
            end
            open(input_name, "w") do io
               write(io, outstr)
            end;
            cmdstr="/beegfs/store/shared/sixs/sixsV2.1 < $input_name > $output_name"
            open(sh_name, "w") do io
               write(io, cmdstr)
            end;
        end
     end


end

