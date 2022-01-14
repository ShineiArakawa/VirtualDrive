import argparse
import carla
import pygame

from API.Carla.core import HUD, KeyboardControl, World


class CarlaAPI:
    def __init__(self) -> None:
        self.__args = None
        self.__world: World = None
        self.__originalSettings = None
        self.__client = None
        self.__simWorld = None
        self.__settings = None
        self.__trafficManager = None
        self.__display = None
        self.__hud = None
        self.__controller = None
        self.__clock = None

        self.__prepare()

    def __prepare(self):
        argparser = argparse.ArgumentParser(
            description='CARLA Manual Control Client')
        argparser.add_argument(
            '-v', '--verbose',
            action='store_true',
            dest='debug',
            help='print debug information')
        argparser.add_argument(
            '--host',
            metavar='H',
            default='127.0.0.1',
            help='IP of the host server (default: 127.0.0.1)')
        argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        argparser.add_argument(
            '-a', '--autopilot',
            action='store_true',
            help='enable autopilot')
        argparser.add_argument(
            '--res',
            metavar='WIDTHxHEIGHT',
            default='1280x720',
            help='window resolution (default: 1280x720)')
        argparser.add_argument(
            '--filter',
            metavar='PATTERN',
            default='vehicle.*',
            help='actor filter (default: "vehicle.*")')
        argparser.add_argument(
            '--generation',
            metavar='G',
            default='2',
            help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
        argparser.add_argument(
            '--rolename',
            metavar='NAME',
            default='hero',
            help='actor role name (default: "hero")')
        argparser.add_argument(
            '--gamma',
            default=2.2,
            type=float,
            help='Gamma correction of the camera (default: 2.2)')
        argparser.add_argument(
            '--sync',
            action='store_true',
            help='Activate synchronous mode execution')
        self.__args = argparser.parse_args()

        self.__args.width, self.__args.height = [
            int(x) for x in self.__args.res.split('x')]

        pygame.init()
        pygame.font.init()
        self.__world: World = None
        self.__originalSettings = None

        self.__client = carla.Client(self.__args.host, self.__args.port)
        self.__client.set_timeout(20.0)

        self.__simWorld = self.__client.get_world()
        if self.__args.sync:
            self.__originalSettings = self.__simWorld.get_settings()
            self.__settings = self.__simWorld.get_settings()
            if not self.__settings.synchronous_mode:
                self.__settings.synchronous_mode = True
                self.__settings.fixed_delta_seconds = 0.05
            self.__simWorld.apply_settings(self.__settings)

            self.__trafficManager = self.__client.get_trafficmanager()
            self.__trafficManager.set_synchronous_mode(True)

        if self.__args.autopilot and not self.__simWorld.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        self.__display = pygame.display.set_mode(
            (self.__args.width, self.__args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.__display.fill((0, 0, 0))
        pygame.display.flip()

        self.__hud = HUD(self.__args.width, self.__args.height)
        self.__world = World(self.__simWorld, self.__hud, self.__args)
        self.__controller = KeyboardControl(
            self.__world, self.__args.autopilot)

        if self.__args.sync:
            self.__simWorld.tick()
        else:
            self.__simWorld.wait_for_tick()

        self.__clock = pygame.time.Clock()

    def drive(self, steeringAngle: float):
        if self.__args.sync:
            self.__simWorld.tick()

        self.__clock.tick_busy_loop(60)

        if self.__controller.parse_events(self.__client, self.__world, self.__clock, self.__args.sync, steeringAngle):
            return

        self.__world.tick(self.__clock)
        self.__world.render(self.__display)
        pygame.display.flip()

    def __del__(self):
        if self.__originalSettings:
            self.__simWorld.apply_settings(self.__originalSettings)

        if (self.__world and self.__world.recording_enabled):
            self.__client.stop_recorder()

        if self.__world is not None:
            self.__world.destroy()

        pygame.quit()
