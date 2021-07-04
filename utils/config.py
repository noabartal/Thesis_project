dataset = 'SHRP2'  # SOROKA
label = 'DriverID'
event = 'eventID'
path_to_data = 'Data'
path_to_results =' Results'
seq_length = 210
step_size = 210

continues_features = ['vtti.accel_x', 'vtti.accel_y', 'vtti.accel_z',
                  'vtti.gyro_x', 'vtti.gyro_y', 'vtti.gyro_z',  # 'vtti.left_marker_probability',
                      # 'vtti.right_marker_probability', 'vtti.left_marker_type', 'vtti.right_marker_type',
                  'vtti.pedal_gas_position', 'vtti.speed_network', 'vtti.light_level']

discrete_features = [  # 'vtti.lane_distance_off_center', 'vtti.lane_width', 'vtti.left_line_right_distance',
    'vtti.pedal_brake_state', 'vtti.prndl',  # 'vtti.right_line_left_distance',
    'vtti.steering_wheel_position', 'vtti.temperature_interior', 'vtti.abs', 'vtti.cruise_state',
    'vtti.headlight', 'vtti.turn_signal', 'vtti.engine_rpm_instant']

time_column = 'vtti.timestamp'

running_porpose = "train_model"  # generate_results

ephocs = 1
SELECTION = ['ufs', 'cfs']

CLASSIFIERS = ['inception', 'inception_extension', 'fcn', 'fcn__extension', 'resnet', 'resnet__extension']
FEATURES = [0.1, 0.2, 0.3]

SELECTION = ['ufs']
DENSE = [32, 64, 'class']
ITERS = 5