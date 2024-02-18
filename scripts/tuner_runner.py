import yaml
import subprocess
YAML_PATH = "/home/mingyo/Projects/jack/TRILL/configs/default.yaml"
START = 1
STEP = 1
END = 500

with open(YAML_PATH, 'r') as file:
    data = yaml.safe_load(file)
    
for i in range(START, END, STEP):
    print("Running with ", i)
    
    # data["Whole-Body Contol"]["kp"]["Torso"][0] = i
    # data["Whole-Body Contol"]["kp"]["Torso"][1] = i
    # data["Whole-Body Contol"]["kp"]["Torso"][2] = i
    # data["Whole-Body Contol"]["kd"]["Torso"][0] = i/10
    # data["Whole-Body Contol"]["kd"]["Torso"][1] = i/10
    # data["Whole-Body Contol"]["kd"]["Torso"][2] = i/10
    
    # data["Whole-Body Contol"]["kp"]["COM"][0] = i
    # data["Whole-Body Contol"]["kp"]["COM"][1] = i
    # data["Whole-Body Contol"]["kp"]["COM"][2] = i
    # data["Whole-Body Contol"]["kd"]["COM"][0] = i/15
    # data["Whole-Body Contol"]["kd"]["COM"][1] = i/15
    # data["Whole-Body Contol"]["kd"]["COM"][2] = i/15
    
    # data["Whole-Body Contol"]["kp"]["Foot Pos"][0] = i
    # data["Whole-Body Contol"]["kp"]["Foot Pos"][1] = i
    # data["Whole-Body Contol"]["kp"]["Foot Pos"][2] = i
    # data["Whole-Body Contol"]["kp"]["Foot Quat"][0] = i
    # data["Whole-Body Contol"]["kp"]["Foot Quat"][1] = i
    # data["Whole-Body Contol"]["kp"]["Foot Quat"][2] = i
    # data["Whole-Body Contol"]["kd"]["Foot Pos"][0] = i/10
    # data["Whole-Body Contol"]["kd"]["Foot Pos"][1] = i/10
    # data["Whole-Body Contol"]["kd"]["Foot Pos"][2] = i/10
    # data["Whole-Body Contol"]["kd"]["Foot Quat"][0] = i/10
    # data["Whole-Body Contol"]["kd"]["Foot Quat"][1] = i/10
    # data["Whole-Body Contol"]["kd"]["Foot Quat"][2] = i/10
    # Write the updated YAML back to disk
    with open(YAML_PATH, 'w') as file:
        yaml.safe_dump(data, file)
    
    
    subprocess.run(['python', '/home/mingyo/Projects/jack/TRILL/scripts/tuner.py', '--parameter', str(i)])
