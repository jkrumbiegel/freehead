from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5 import QtWidgets
import vispy.scene
import vispy.io
import vispy.app

vispy.use('pyqt5')
from vispy.scene import visuals
from vispy.visuals.transforms import MatrixTransform
from vispy.visuals.filters import Alpha
from vispy.gloo.util import _screenshot
import pandas as pd
import freehead as fh
import os
import numpy as np
from collections import OrderedDict
from PIL import Image


def create_4x4_matrix(R, T, scale):
    M = np.diag(np.array([0, 0, 0, 1.0]))
    M[:3, :3] = R * scale
    M[:3, 3] = T
    return M.astype(np.float32).T


class ExperimentVisualization(QtWidgets.QWidget):
    trial_changed = pyqtSignal(int)
    frame_changed = pyqtSignal(int)

    def __init__(self, exp_df_path, trial_df_path, rig_leds_path):

        exp_df = pd.read_pickle(exp_df_path)
        trial_df = pd.read_pickle(trial_df_path)
        self.df = exp_df.join(trial_df.drop('block', axis=1), on='trial_number')
        self.rig_leds = np.load(rig_leds_path)

        self.precalculate_data()

        verts, faces, normals, nothin = vispy.io.read_mesh(os.path.join(fh.PACKAGE_DIR, '../datafiles', 'head.obj'))
        verts = np.einsum('ni,ji->nj', (verts - verts.mean(axis=0)), fh.from_yawpitchroll(180, 90, 0))
        #verts = verts - verts.mean(axis=0)

        # add SceneCanvas first to create QApplication object before widget
        self.vispy_canvas = vispy.scene.SceneCanvas(create_native=True, vsync=True, show=True, bgcolor=(0.2, 0.2, 0.2, 0))

        super(ExperimentVisualization, self).__init__()

        #self.setAttribute(Qt.WA_TranslucentBackground)

        self.timer = vispy.app.Timer(1 / 30, start=False, connect=self.advance_frame)

        self.n_trials = len(self.df)
        self.current_trial = 0
        self.i_frame = 0
        self.current_row = self.df.iloc[self.current_trial]
        self.current_R_helmet = None
        self.current_gaze_normals = None
        self.current_ref_points = None

        self.vispy_view = self.vispy_canvas.central_widget.add_view()
        self.vispy_view.camera = 'turntable'
        self.vispy_view.camera.center = self.rig_leds[127, :] + (self.rig_leds[0, :] - self.rig_leds[127, :]) + (
                    self.rig_leds[254, :] - self.rig_leds[127, :])
        self.vispy_view.camera.fov = 40
        self.vispy_view.camera.distance = 1500

        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)

        self.frame_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.on_slider_change)

        self.trial_picker = QtWidgets.QSpinBox()
        self.trial_picker.setMaximum(self.n_trials)
        self.trial_picker.valueChanged.connect(self.on_picker_change)

        self.trial_changed.connect(self.load_trial)
        self.frame_changed.connect(self.load_frame)

        self.picker_slider_layout = QtWidgets.QHBoxLayout()

        self.main_layout.addWidget(self.vispy_canvas.native)
        self.main_layout.addLayout(self.picker_slider_layout)

        self.animation_button = QtWidgets.QPushButton('Start Animation')
        self.picker_slider_layout.addWidget(self.animation_button)
        self.animation_button.clicked.connect(self.toggle_animation)

        self.frame_label = QtWidgets.QLabel('Frame')
        self.picker_slider_layout.addWidget(self.frame_label)
        self.picker_slider_layout.addWidget(self.frame_slider)
        self.picker_slider_layout.addWidget(QtWidgets.QLabel('Trial'))
        self.picker_slider_layout.addWidget(self.trial_picker)

        self.rig_vis = visuals.Markers()
        self.rig_vis.set_gl_state(depth_test=False)
        self.rig_vis.antialias = 0
        self.vispy_view.add(self.rig_vis)

        self.helmet_vis = visuals.Markers()
        self.vispy_view.add(self.helmet_vis)

        self.gaze_vis = visuals.Line()
        self.vispy_view.add(self.gaze_vis)

        self.head_mesh = visuals.Mesh(vertices=verts, shading='smooth', faces=faces, mode='triangles',
                                      color=(0.5, 0.55, 0.7))
        self.head_mesh.shininess = 0
        self.head_mesh.light_dir = [0, 1, 1]
        # self.head_mesh.light_color = np.array((1, 1, 0.95)) * 0.8
        self.head_mesh.ambient_light_color = np.array((0.98, 0.98, 1)) * 0.2
        self.head_mesh.attach(Alpha(0.3))
        self.head_mesh.set_gl_state(depth_test=True, cull_face=True)
        self.head_mesh_transform = MatrixTransform()
        self.head_mesh.transform = self.head_mesh_transform
        self.vispy_view.add(self.head_mesh)

        self.trial_changed.emit(0)
        self.frame_changed.emit(0)

        self.show()
        vispy.app.run()

    def toggle_animation(self):
        if self.timer.running:
            self.timer.stop()
            self.animation_button.setText('Start Animation')
        else:
            self.timer.start()
            self.animation_button.setText('Stop Animation')

    def advance_frame(self, t):
        new_frame = (self.i_frame + 4) % self.current_gaze_normals.shape[0]
        self.frame_changed.emit(new_frame)

    def on_picker_change(self):
        self.trial_changed.emit(self.trial_picker.value())

    def on_slider_change(self):
        self.frame_changed.emit(self.frame_slider.value())

    @pyqtSlot(int)
    def load_trial(self, i):
        if i >= self.n_trials:
            raise ValueError(f'{i} is too big an index for trial data')
        self.current_row = self.df.iloc[i]
        self.current_R_helmet = self.current_row['R_head_world']
        self.current_gaze_normals = self.current_row['gaze_normals']
        self.current_ref_points = self.current_row['Ts_head_world']

        self.frame_slider.setMinimum(0)
        n_frames = self.current_gaze_normals.shape[0]
        self.frame_slider.setMaximum(n_frames - 1)
        new_frame = np.clip(self.i_frame, 0, n_frames)
        self.frame_changed.emit(new_frame)

        self.helmet_vis.set_data(self.current_ref_points[self.i_frame, :], face_color=(0, 1, 0), size=5)

    @pyqtSlot(int)
    def load_frame(self, i):
        with self.vispy_canvas.events.blocker():
            self.i_frame = i

            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(i)
            self.frame_slider.blockSignals(False)

            self.frame_label.setText(f'Frame: {i:5d}\t')

            ref_points = self.current_ref_points[i, :]
            self.helmet_vis.set_data(ref_points, face_color=(0, 1, 0), size=5)

            gaze_start = ref_points[5, :]
            gaze_end = gaze_start + 2000 * self.current_row['gaze_world'][i, ...]
            self.gaze_vis.set_data(np.vstack((gaze_start, gaze_end)))

            head_centroid = ref_points[1:5, :].mean(axis=0)
            self.head_mesh_transform.matrix = create_4x4_matrix(
                self.current_R_helmet[i, ...],
                head_centroid,
                np.array([85, 90, 95]))

            self.update_rig()

    def precalculate_data(self):
        fh.array_apply(
            self.df,
            OrderedDict([
                ('target_led', lambda r: r['fixation_led'] + r['amplitude']),
                ('shifted_target_led', lambda r: r['target_led'] + r['shift']),
                # optotrak data interpolated onto pupil labs timestamps
                ('o_data_interp',
                 lambda r: fh.interpolate_a_onto_b_time(r['o_data'][:, 3:15], r['o_data'][:, 30], r['p_data'][:, 2],
                                                     kind='linear')),
                # gaze data
                ('gaze_normals', lambda r: r['p_data'][:, 3:6]),
                # rotation of head rigidbody
                ('R_head_world', lambda r: r['helmet'].solve(r['o_data_interp'].reshape((-1, 4, 3)))[0]),
                # yaw pitch roll head rigidbody
                ('ypr_head_world', lambda r: fh.to_yawpitchroll(r['R_head_world'])),
                # reference positions of head rigidbody
                ('Ts_head_world', lambda r: r['helmet'].solve(r['o_data_interp'].reshape((-1, 4, 3)))[1]),
                # position of target led
                ('target_pos', lambda r: self.rig_leds[r['target_led'], :]),
                # vector from eye to target position
                ('eye_to_target', lambda r: fh.to_unit(r['target_pos'] - r['Ts_head_world'][:, 5, :])),
                # gaze vector in head without distortion correction
                ('gaze_head_distorted', lambda r: (r['R_eye_head'] @ r['gaze_normals'].T).T),
                # gaze vector in head with distortion correction
                ('gaze_head',
                 lambda r: fh.normals_nonlinear_angular_transform(r['gaze_head_distorted'], r['nonlinear_parameters'])),
                # gaze vector in world
                ('gaze_world', lambda r: np.einsum('tij,tj->ti', r['R_head_world'], r['gaze_head'])),
                # gaze angles in world
                ('gaze_ang_world', lambda r: np.rad2deg(fh.to_azim_elev(r['gaze_world']))),
                # angles from eye to target in world
                ('eye_ang_target', lambda r: np.rad2deg(fh.to_azim_elev(r['eye_to_target']))),
                # difference of eye to target angles and gaze in world
                ('d_ang_gaze_eye_target', lambda r: r['gaze_ang_world'] - r['eye_ang_target']),
                # time steps
            ]),
            add_inplace=True
        )

    def update_rig(self):

        fix_led = self.current_row['fixation_led']
        target_led = self.current_row['target_led']
        shifted_led = self.current_row['shifted_target_led']

        rig_color_neutral = np.array([0.5, 0.5, 0.5, 0.5])
        rig_color_fix = np.array([1, 0, 0, 1])
        rig_color_target = np.array([0, 1, 0, 1])
        rig_color_shifted = np.array([1, 1, 0, 1])

        rig_colors = np.tile(rig_color_neutral, (255, 1))
        rig_colors[fix_led, :] = rig_color_fix
        rig_colors[target_led, :] = rig_color_target
        rig_colors[shifted_led, :] = rig_color_shifted

        bigger_size = 10
        sizes = np.ones(255, dtype=int) * 5

        if self.i_frame < self.current_row['i_started_fixating']:
            pass
        elif self.current_row['i_started_fixating'] <= self.i_frame < self.current_row['i_target_appeared']:
            sizes[fix_led] = bigger_size
        elif self.current_row['i_target_appeared'] <= self.i_frame < self.current_row['i_saccade_started']:
            sizes[target_led] = bigger_size
        elif self.current_row['i_saccade_started'] <= self.i_frame < self.current_row['i_saccade_landed']:
            sizes[shifted_led] = bigger_size
        else:
            pass

        self.rig_vis.set_data(self.rig_leds, face_color=rig_colors, edge_color=None, size=sizes)
