import pygame
import sys

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("Not Connected")
    sys.exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Connect Success：{joystick.get_name()}")
print(f"- Button Number: {joystick.get_numbuttons()}")
print(f"- Joystick Number: {joystick.get_numaxes()}")
print(f"- Hat（D-Pad）Number: {joystick.get_numhats()}")
print("Start monitor input...（Ctrl+C quit）\n")

while True:
    for event in pygame.event.get():
        if event.type == pygame.JOYBUTTONDOWN:
            print(f"[button press]  ID: {event.button}")
        elif event.type == pygame.JOYBUTTONUP:
            print(f"[button release]  ID: {event.button}")
        elif event.type == pygame.JOYAXISMOTION:
            if abs(event.value) > 0.2:
                print(f"[joystick]  axis {event.axis} value: {event.value:.2f}")
        elif event.type == pygame.JOYHATMOTION:
            print(f"[D-Pad]  orientation: {event.value}")
        elif event.type == pygame.JOYDEVICEADDED:
            print("joystick connected")
        elif event.type == pygame.JOYDEVICEREMOVED:
            print("joystick disconnected")
