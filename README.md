# LumenSense AI using YOLOv8
## Introduction
The Smart Occupancy Lighting System is an innovative solution that leverages computer vision and machine learning to automate lighting control in indoor spaces. Using YOLOv8, a state-of-the-art object detection model, this system accurately detects and counts occupants in real-time, adjusting lighting accordingly to optimize energy usage.
## Features

* Real-time person detection using YOLOv8
* Accurate occupancy counting
* Automated lighting control based on room occupancy
* Energy consumption monitoring and reporting
* Configurable sensitivity and delay settings

## Technology Stack

* Python
* OpenCV for video capture and processing
* YOLOv8 for object detection
* Maths
* Tracking Algorithm
* PyTorch (YOLOv8 backend)
* Raspberry Pi (optional for deployment)

## Workings
1. Takes input as a Video, Read it frame by frame and detect the objects.
2. In each frame the object ID is alloted to a centroid of that object, When switching to next frame similarity algorithm detect the similar object, if found update new centroid, if not delete that centroid and object ID.
3. When the centroid of object passes the line specified and boundary of room entery if adds or removes from memory according as it enters or exits.
4. If centroid enters it increase number of peoples and as it exits it decreases.

## Usage 
* It help to save electricity in a household or organizationak institute by switching of light of area of no use for a threshold period.
* Help in Public places to know how many peoples are stuck in case of some calamities.
* Help for costumer data analysis and buisness plannings.

## How It Works

* Video Input: Captures real-time video feed from a camera
* Object Detection: Processes frames using YOLOv8 to detect people
* Occupancy Counting: Tracks the number of people in the room
* Lighting Control: Manages lights based on occupancy status
* Data Logging: Records occupancy and energy usage data

## Future Improvements

* Integration with smart home systems
* Multi-room support
* Advanced analytics for occupancy patterns

## Conclusion
The Smart Occupancy Lighting System using YOLOv8 demonstrates the powerful intersection of computer vision, machine learning, and energy management. By automating lighting control based on real-time occupancy detection, this project offers a practical solution for reducing energy waste in various settings, from office spaces to residential buildings.

### Key takeaways:

* Showcases practical application of advanced AI in everyday scenarios
* Promotes energy efficiency and cost savings
* Provides a scalable foundation for smart building management
* Offers hands-on experience with cutting-edge technologies like YOLOv8\

This project not only serves as a testament to the potential of AI in creating sustainable environments but also provides a valuable learning experience in integrating complex systems for real-world applications. As smart building technologies continue to evolve, solutions like this will play a crucial role in shaping more efficient and responsive spaces.
