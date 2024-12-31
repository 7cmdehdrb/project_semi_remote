#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Bool, Int32MultiArray
from onrobot import VG

class OnRobotDemo:
    def __init__(self):
        rospy.init_node('vacuum_demo')
        self.vg = VG("192.168.1.1", "502")
        self.sub_target = rospy.Subscriber('/vacuum', String, self.vacuum_callback)
        self.sub_target_reached = rospy.Subscriber('/target_reached', Bool, self.target_reached_callback)
        self.sub_placement_reached = rospy.Subscriber('/placement_reached', Bool, self.placement_reached_callback)
        self.pub_vacuum_status = rospy.Publisher('/vacuum_status', Int32MultiArray, queue_size=10)

    def vacuum_callback(self, msg):
        if msg.data == "on":
            self.vg.vacuum_on()
            status = Int32MultiArray()
            status.data = [self.vg.get_channelA_vacuum(), self.vg.get_channelB_vacuum()]
            self.pub_vacuum_status.publish(status)
        elif msg.data == "off":
            self.vg.release_vacuum()
            status = Int32MultiArray()
            status.data = [0, 0]  # Assuming 0 represents the "off" state for vacuum channels
            self.pub_vacuum_status.publish(status)
        else:
            print("error")

    def target_reached_callback(self, msg):
        # print(msg.data)
        if msg.data is True:
            self.vg.vacuum_on()
            print("vacuum on")
            
        elif msg.data is False:
            self.vg.release_vacuum()
            print("vacuum off")
                
    def placement_reached_callback(self, msg):
        # print(msg.data)
        if msg.data == False:
            self.vg.release_vacuum()
            
if __name__ == '__main__':
    onrobotdemo = OnRobotDemo()
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        status = Int32MultiArray()
        status.data = [onrobotdemo.vg.get_channelA_vacuum(), onrobotdemo.vg.get_channelB_vacuum()]
        onrobotdemo.pub_vacuum_status.publish(status)
        rate.sleep()