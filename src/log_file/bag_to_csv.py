import os
import sys


def ros_command(bag_file: str, topic_name: str, output_path: str):
    if not "/" in topic_name:
        topic_name = "/" + topic_name

    output_file = topic_name.replace("/", "") + ".csv"
    output_file = output_path + "/" + output_file

    return f"rostopic echo -b {bag_file} -p {topic_name} > {output_file}"


bag_file = "/home/workspace/src/log_file/0106.bag"

topics = [
    "/box_objects/pdf",
    "/control_mode",
    "/desired_box",
    "/intention/distance",
    "/intention/intersection_length",
    "/intention/log/gaussian",
]

for topic_name in topics:
    command = ros_command(bag_file, topic_name, "/home/workspace/src/log_file/raw_1")
    print(command)
    os.system(command)
    print(f"{topic_name} is saved as csv file")
