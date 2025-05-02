class Enums:

    mapping_dict = {
        'Car': 'Car',
        'Van': 'Car',
        'Truck': 'Car',
        'Pedestrian': 'Pedestrian',
        'Person_sitting': 'Pedestrian',
        'Cyclist': 'Cyclist'
    }
    
    KiTTi_label2Id = {
        'Car': 0,
        'Cyclist': 1,
        'Pedestrian': 2
    }

    KiTTi_Id2label = {
        0: 'Car',
        1: 'Cyclist',
        2: 'Pedestrian',
    }
    
    KiTTi_class_colors = {
        0: (255, 0, 0),  # Red (class 0)
        1: (0, 255, 0),  # Green (class 1)
        2: (0, 0, 255)   # Blue (class 2)
    }    

    reduced_KiTTi_label2Id = {
        'Car': 0,
        'Van': 0,
        'Truck': 0,
        'Pedestrian': 3,
        'Person_sitting': 3,
        'Cyclist': 1,
        'Tram': 2,
        'Misc': 2,
        'DontCare': 2
    }

    KiTTi_label2Id_SSD = {
        'Background':0,
        'Car': 1,
        'Cyclist': 2,
        'Pedestrian': 3
    }

    KiTTi_Id2label_SSD = {
        0: 'Background',
        1: 'Car',
        2: 'Cyclist',
        3: 'Pedestrian',
    }
    
    # Map fine-grained nuScenes labels to general object classes
    NUSCENES_TO_GENERAL_CLASSES = {
        # Vehicles
        "vehicle.car": "vehicle",
        "vehicle.truck": "vehicle",
        "vehicle.bus.rigid": "vehicle",
        "vehicle.bus.bendy": "vehicle",
        "vehicle.trailer": "vehicle",
        "vehicle.construction": "vehicle",
        "vehicle.emergency.police": "vehicle",

        # Cycles
        "vehicle.bicycle": "cycle",
        "vehicle.motorcycle": "cycle",

        # Pedestrians
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "human.pedestrian.stroller": "pedestrian",
        "human.pedestrian.wheelchair": "pedestrian",
        "human.pedestrian.personal_mobility": "pedestrian",

        # Movable objects
        "movable_object.barrier": "barrier",
        "movable_object.trafficcone": "traffic_cone",
        "movable_object.debris": "movable_object",
        "movable_object.pushable_pullable": "movable_object",

        # Static
        "static_object.bicycle_rack": "static_object",

        # Animals
        "animal": "animal"
    }

    nuscenes_label2Id = {
        "vehicle": 0,
        "cycle": 1,
        "pedestrian": 2,
        # "barrier": 3,
        # "traffic_cone": 4,
        # "movable_object": 5,
        # # "static_object": 6,
        # # "animal": 7
    }

    nuscenes_Id2Labels = {
        0: "vehicle",
        1: "cycle",
        2: "pedestrian",
        3: "barrier",
        4: "traffic_cone",
        5: "movable_object",
        6: "static_object",
        7: "animal"
    }

    # KiTTi_label2Id = {
    #     'Car': 0,
    #     'Van': 1,
    #     'Truck': 2,
    #     'Pedestrian': 3,
    #     'Person_sitting': 4,
    #     'Cyclist': 5,
    #     'Tram': 6,
    #     'Misc': 7,
    #     'DontCare': 8
    # } 
    
    # KiTTi_Id2label = {
    #     0: 'Car',
    #     1: 'Van',
    #     2: 'Truck',
    #     3: 'Pedestrian',
    #     4: 'Person_sitting',
    #     5: 'Cyclist',
    #     6: 'Tram',
    #     7: 'Misc',
    #     8: 'DontCare'
    # }