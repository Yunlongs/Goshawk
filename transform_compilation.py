import json
import sys
path = sys.argv[1]
with open(path + 'compile_commands.json', 'r') as f:
    data = json.load(f)
output = []
for item in data:
    command = " ".join(item['arguments'])
    directory = item['directory']
    file = item['file']
    output_dict = {
        "directory": directory,
        "command": command + " -fcolor-diagnostics",
        "file": file
    }
    output.append(output_dict)
print(output)
with open(path+'compilation.json', 'w') as f:
    json.dump(output, f)
