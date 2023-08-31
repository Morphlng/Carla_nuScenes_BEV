import argparse
import random
import sys

import carla
import numpy as np
import pygame

from birdeye_render import BirdeyeRender


class SimpleEnv:
    def __init__(self, args):
        self.args = args

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        if self.world.get_map().name.split('/')[-1] != 'Town05':
            self.world = self.client.load_world("Town05")
        self.map = self.world.get_map()

        # display and render related
        pygame.init()
        if args.no_render:
            self.display = pygame.display.set_mode((500, 500), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.HIDDEN)
        else:
            self.display = pygame.display.set_mode((500, 500), pygame.HWSURFACE | pygame.DOUBLEBUF)
        
        birdeye_params = {
            'screen_size': [500, 500],
            'pixels_per_meter': 8,
            'pixels_ahead_vehicle': 8 * (32/2 - 12) # (obs_range/2 - d_behind) * pixels_per_meter
        }
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)
    
    def setup_scenario(self):
        # Spawn an Ego and autopilot
        actors = self.world.get_actors()
        self.ego = None

        for actor in actors:
            if actor.attributes.get('role_name') == 'hero':
                self.ego = actor
                break

        if self.ego is None:
            print("Ego not found, spawning one...")
            ego_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
            ego_bp.set_attribute('role_name', 'hero')
            spawn_point = random.choice(self.map.get_spawn_points())
            self.ego = self.world.spawn_actor(ego_bp, spawn_point)
            self.ego.set_autopilot(True)
        
        self.birdeye_render.set_hero(self.ego, self.ego.id)
        
        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        self.full_road_waypoints = self.generate_road_waypoints()
    
    def generate_road_waypoints(self):
        """Return all, precisely located waypoints from the map.

        Topology contains simplified representation (a start and an end
        waypoint for each road segment). By expanding each until another
        road segment is found, we explore all possible waypoints on the map.

        Returns a list of waypoints for each road segment.
        """
        precision = 0.05
        road_segments_starts: carla.Waypoint = [
            road_start for road_start, road_end in self.map.get_topology()
        ]
        
        def extract_waypoint(wpt):
            return (wpt.transform.location.x, wpt.transform.location.y, wpt.transform.rotation.yaw)

        each_road_waypoints = []
        for road_start_waypoint in road_segments_starts:
            road_waypoints = [extract_waypoint(road_start_waypoint)]

            # Generate as long as it's the same road
            next_waypoints = road_start_waypoint.next(precision)

            if len(next_waypoints) > 0:
                # Always take first (may be at intersection)
                next_waypoint = next_waypoints[0]
                while next_waypoint.road_id == road_start_waypoint.road_id:
                    road_waypoints.append(extract_waypoint(next_waypoint))
                    next_waypoint = next_waypoint.next(precision)

                    if len(next_waypoint) > 0:
                        next_waypoint = next_waypoint[0]
                    else:
                        # Reached the end of road segment
                        break
            each_road_waypoints.append(road_waypoints)
        return each_road_waypoints
    
    def step(self):
        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > 3:
            self.vehicle_polygons.pop(0)
        
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > 3:
            self.walker_polygons.pop(0)
    
    def render(self):
        ## Birdeye rendering
        self.birdeye_render.vehicle_polygons = self.vehicle_polygons
        self.birdeye_render.walker_polygons = self.walker_polygons
        self.birdeye_render.waypoints = self.nearby_waypoints()

        # birdeye view with roadmap, actors and waypoints
        birdeye_render_types = ['roadmap', 'actors', 'waypoints']
        self.birdeye_render.render(self.display, birdeye_render_types)
        if self._event_handler():
            pygame.quit()
            sys.exit()
            
        return pygame.surfarray.array3d(self.display).swapaxes(0, 1)

    def close(self):
        self.client = None
        pygame.quit()
    
    def _event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return True
        return False
    
    def nearby_waypoints(self):
        """Return only the road that is nearby the ego vehicle."""
        nearby_waypoints = []
        render_x, render_y = self.birdeye_render.params['screen_size']
        pixels_per_meter = self.birdeye_render.params['pixels_per_meter']

        half_width = render_x / (2 * pixels_per_meter)
        half_height = render_y / (2 * pixels_per_meter)

        ego_x = self.ego.get_location().x
        ego_y = self.ego.get_location().y

        # Define bounding box
        left_bound = ego_x - half_width
        right_bound = ego_x + half_width
        top_bound = ego_y - half_height
        bottom_bound = ego_y + half_height

        for road_waypoints in self.full_road_waypoints:
            road = [waypoint for waypoint in road_waypoints if left_bound < waypoint[0] < right_bound and top_bound < waypoint[1] < bottom_bound]
            
            if road:
                nearby_waypoints.append(road)

        return nearby_waypoints

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
        filt: the filter indicating what type of actors we'll look at.

        Returns:
        actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict={}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor\
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l,w], [l,-w], [-l,-w], [-l,w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw),-np.sin(yaw)], [np.sin(yaw),np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R,poly_local).transpose() + np.repeat([[x,y]],4,axis=0)
            actor_poly_dict[actor.id]=poly
        return actor_poly_dict


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Carla birdeye view script")
    argparser.add_argument("--host", default="localhost", help="IP of the host server (default: localhost)")
    argparser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
    argparser.add_argument("--no-render", default=False, action="store_true", help="Whether to render pygame window or not")
    args = argparser.parse_args()
    
    
    env = SimpleEnv(args)
    env.setup_scenario()
    try:
        while True:
            env.step()
            env.render()
            pygame.display.flip()
            pygame.time.wait(33)
    finally:
        env.close()
