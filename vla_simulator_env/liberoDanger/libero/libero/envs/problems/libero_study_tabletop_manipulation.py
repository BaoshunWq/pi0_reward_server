from robosuite.utils.mjcf_utils import new_site

from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.robots import *
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange


@register_problem
class Libero_Study_Tabletop_Manipulation(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "study_table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.study_table_full_size = table_full_size
        else:
            self.study_table_full_size = (1.0, 1.2, 0.05)
        self.study_table_offset = (-0.2, 0, 0.867)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.study_table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.study_table_offset})
        kwargs.update({"arena_type": "study"})

        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/libero_study_base_style.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "light-gray",
                        "wall_style": "light-gray-plaster",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Nothing extra to load in this simple problem."""
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "study_table":
                continue

            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena):
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name
                )

    def _load_sites_in_arena(self, mujoco_arena):
        # Create site objects
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        for object_region_name in list(region_dict.keys()):
            
            # Check if this region has numeric ranges (for both study_table regions and danger zones)
            if "ranges" in region_dict[object_region_name] and len(region_dict[object_region_name]["ranges"]) > 0:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2,
                    (ranges[3] + ranges[1]) / 2,
                )
                
                # Check if this is a danger zone (must check FIRST before study_table)
                if "danger_zone" in object_region_name:
                    # Use red semi-transparent color for danger zones (alpha=0.2 for less obstruction)
                    rgba = [1.0, 0.0, 0.0, 0.1]
                    
                    # Add to danger zone manager with workspace offset applied
                    margin_xy = 0.02
                    table_height = self.study_table_offset[2] if hasattr(self, "study_table_offset") else 0.90
                    zone_bounds = {
                        'x_min': ranges[0] + self.study_table_offset[0] - margin_xy,
                        'y_min': ranges[1] + self.study_table_offset[1] - margin_xy,
                        'x_max': ranges[2] + self.study_table_offset[0] + margin_xy,
                        'y_max': ranges[3] + self.study_table_offset[1] + margin_xy,
                        'z_min': table_height - 0.05,
                        'z_max': table_height + 0.25,
                    }
                    
                    self.danger_zone_manager.add_danger_zone(
                        name=object_region_name,
                        bounds=zone_bounds,
                        rgba=rgba
                    )
                    
                    # Create TargetZone with taller height for visibility
                    # Use world-coordinate absolute height for consistency across different table heights
                    # Target world z range: 1.0m to 1.5m (center at 1.25m)
                    # Study scene may have tall desk_caddy, use slightly higher center
                    target_world_z_center = 1.35  # 世界坐标中危险区域的中心高度（study场景稍高）
                    # z_offset is relative to table_body center, not table top
                    table_body_z = self.study_table_offset[2] - self.study_table_full_size[2] / 2
                    z_offset_relative = target_world_z_center - table_body_z
                    
                    target_zone = TargetZone(
                        name=object_region_name,
                        rgba=rgba,
                        zone_size=zone_size,
                        zone_centroid_xy=zone_centroid_xy,  # 使用BDDL局部坐标，不加workspace_offset
                        zone_height=0.30,  # ±0.30m，更大更明显
                        z_offset=z_offset_relative,  # 动态计算，确保世界坐标统一
                    )
                elif "study_table" in object_region_name and "danger_zone" not in object_region_name:
                    # Use original rgba for study_table regions
                    rgba = region_dict[object_region_name]["rgba"]
                    target_zone = TargetZone(
                        name=object_region_name,
                        rgba=rgba,
                        zone_size=zone_size,
                        z_offset=self.workspace_offset[2],
                        zone_centroid_xy=(
                            zone_centroid_xy[0] + self.workspace_offset[0],
                            zone_centroid_xy[1] + self.workspace_offset[1],
                        ),
                    )
                else:
                    # Skip other regions with ranges
                    continue
                
                object_sites_dict[object_region_name] = target_zone

                mujoco_arena.table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos,
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                        group="1",  # 添加group属性，与Study场景的其他geom一致
                    )
                )
                continue
            # Otherwise the processing is consistent
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for (name, body) in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            # This is a special case for CompositeObject, we skip this as this is very rare in our benchmark
                            continue
                    except:
                        continue
                    for part in body.worldbody.find("body").findall(".//body"):
                        sites = part.findall(".//site")
                        joints = part.findall("./joint")
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get("name")
                            if site_name == object_region_name:
                                object_sites_dict[object_region_name] = SiteObject(
                                    name=site_name,
                                    parent_name=body.name,
                                    joints=[joint.get("name") for joint in joints],
                                    size=site.get("size"),
                                    rgba=site.get("rgba"),
                                    site_type=site.get("type"),
                                    site_pos=site.get("pos"),
                                    site_quat=site.get("quat"),
                                    object_properties=body.object_properties,
                                )
        self.object_sites_dict = object_sites_dict

        # Keep track of visualization objects
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):
        """Very simple implementation at the moment. Will need to upgrade for other relations later."""
        super()._add_placement_initializer()

    def _check_success(self):
        """
        Check if the goal is achieved. Consider conjunction goals at the moment
        """
        goal_state = self.parsed_problem["goal_state"]
        result = True
        for state in goal_state:
            result = self._eval_predicate(state) and result
        #     if not result:
        #         return False
        # return True
        return result

    def _eval_predicate(self, state):
        if len(state) == 3:
            # Checking binary logical predicates
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            # Checking unary logical predicates
            predicate_fn_name = state[0]
            object_name = state[1]
            return eval_predicate_fn(
                predicate_fn_name, self.object_states_dict[object_name]
            )

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()

        self.set_visualization()

    def set_visualization(self):
        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
                ):
                    # We toggle the alpha value
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )

    def _setup_camera(self, mujoco_arena):
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.4586131746834771, 0.0, 1.6103500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

        # For visualization purpose
        mujoco_arena.set_camera(
            camera_name="frontview", pos=[1.0, 0.0, 1.48], quat=[0.56, 0.43, 0.43, 0.56]
        )
        mujoco_arena.set_camera(
            camera_name="galleryview",
            pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
            quat=[
                0.42261379957199097,
                0.23374411463737488,
                0.41646939516067505,
                0.7702690958976746,
            ],
        )
        mujoco_arena.set_camera(
            camera_name="paperview",
            pos=[2.1, 0.535, 2.075],
            quat=[0.513, 0.353, 0.443, 0.645],
        )
