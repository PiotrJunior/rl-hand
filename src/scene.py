# scene_builder.py
import genesis as gs

def build_orca_scene(show_viewer: bool):
    """
    Initializes the Genesis engine and constructs the physical world simulation.
    
    Args:
        show_viewer (bool): If True, opens a 3D GUI window for visual debugging.
                            Set to False for headless LeRobot training.
                            
    Returns:
        tuple: (gs.Scene, dict) containing the compiled scene object and a 
               dictionary of references to all instantiated entities.
    """
    try:
        gs.init(backend=gs.gpu)
    except gs.GenesisException:
        # Catch exception if the engine is accidentally re-initialized within the same process
        pass
        
    scene = gs.Scene(
        # sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1, 1, 2),
            camera_lookat=(0, 0, 1),
            res=(1280, 720),
            max_FPS=60,
        ),
        show_viewer=show_viewer
    )
    
    # 1. Environment Ground Plane
    plane = scene.add_entity(gs.morphs.Plane())
    
    # 2. Red Sphere (The target object that needs to be pushed/manipulated)
    object = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.3, 0, 0.2), radius=0.08),
        surface=gs.surfaces.Default(color=(0.5, 0.0, 0.0))
    )
    
    # 3. Green Box
    box = scene.add_entity(
        morph=gs.morphs.Box(pos=(-.25, 0, 0.025), size=(0.3, 0.3, 0.05), fixed=True),
    )
    
    # Hand
    hand = scene.add_entity(
    gs.morphs.URDF(
        file='orcahand_description/v2/models/urdf/orcahand_right.urdf',
        pos=(0.3, -0.3, 0.5),
        collision=True,
        recompute_inertia=True
    ),
)
    
    # 5. Virtual Camera
    camera = scene.add_camera(
        res=(480, 480),
        pos=(0, -1, 1),
        lookat=(0, 0, 0.1),
        fov=45,
        GUI=True,
    )

    scene.build()
    
    entities = {
        "plane": plane,
        "object": object,
        "box": box,
        "hand": hand,
        "camera": camera
    }
    
    return scene, entities