import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QPushButton, QLabel,
    QTextEdit, QLineEdit, QFileDialog, QGroupBox, QVBoxLayout
)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
import re


class NASApp(QWidget):

    def __init__(self):
        super().__init__()

        self.dataset_path = None

        self.setWindowTitle("Micro NAS Engine")
        self.setWindowIcon(QIcon("bot.png"))
        self.resize(1100, 700)

        layout = QGridLayout(self)

        # ---------------- DATASET PANEL ----------------

        dataset_box = QGroupBox("Dataset Selection")
        dataset_layout = QVBoxLayout()

        self.dataset_label = QLabel("No dataset selected")

        self.browse_btn = QPushButton("Browse Dataset")
        self.browse_btn.clicked.connect(self.browse_dataset)

        self.dataset_ok_btn = QPushButton("OK")
        self.dataset_ok_btn.clicked.connect(self.load_dataset)

        dataset_layout.addWidget(self.dataset_label)
        dataset_layout.addWidget(self.browse_btn)
        dataset_layout.addWidget(self.dataset_ok_btn)

        dataset_box.setLayout(dataset_layout)

        # ---------------- TRAINING LOG PANEL ----------------

        log_box = QGroupBox("Training Log")

        log_layout = QVBoxLayout()

        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)

        log_layout.addWidget(self.training_log)

        log_box.setLayout(log_layout)

        # ---------------- COMMAND PANEL ----------------

        command_box = QGroupBox("Commands")

        command_layout = QVBoxLayout()

        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText(
            "Example: train 3 gens with 15 models each"
        )

        self.run_command_btn = QPushButton("Run Command")
        self.run_command_btn.clicked.connect(self.handle_command)

        command_layout.addWidget(self.command_input)
        command_layout.addWidget(self.run_command_btn)

        command_box.setLayout(command_layout)

        # ---------------- OUTPUT PANEL ----------------

        output_box = QGroupBox("Output")

        output_layout = QVBoxLayout()

        self.output_panel = QTextEdit()
        self.output_panel.setReadOnly(True)

        output_layout.addWidget(self.output_panel)

        output_box.setLayout(output_layout)

        # ---------------- GRID LAYOUT ----------------

        layout.addWidget(dataset_box, 0, 0)
        layout.addWidget(log_box, 0, 1)
        layout.addWidget(command_box, 1, 0)
        layout.addWidget(output_box, 1, 1)

    # ---------------- DATASET FUNCTIONS ----------------

    def browse_dataset(self):

        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")

        if path:
            self.dataset_path = path
            self.dataset_label.setText(path)

    def load_dataset(self):

        if not self.dataset_path:
            self.log("No dataset selected.")
            return

        self.log(f"Loading dataset from: {self.dataset_path}")

        steps = [
            "Scanning dataset files...",
            "Building dataloader...",
            "Preparing training pipeline...",
            "Dataset ready."
        ]

        self.run_steps(steps)

    # ---------------- COMMAND HANDLING ----------------

    def handle_command(self):

        text = self.command_input.text().strip()

        if not text:
            return

        self.command_input.clear()

        cmd = text.lower()

        train_match = re.search(r"(\d+)\s*gen.*?(\d+)\s*model", cmd)

        if train_match:

            gens = int(train_match.group(1))
            models = int(train_match.group(2))

            self.run_nas(gens, models)
            return

        compare_match = re.search(r"compare.*?(\d+).*?(\d+)", cmd)

        if compare_match:

            g1 = int(compare_match.group(1))
            g2 = int(compare_match.group(2))

            self.compare_generations(g1, g2)
            return

        self.output("Unknown command.")

    # ---------------- TRAINING ----------------

    def run_nas(self, generations, models):

        self.log(f"Starting NAS with {generations} generations.")

        steps = []

        for g in range(1, generations + 1):
            steps.append(f"Training generation {g}")
            steps.append(f"Evaluating {models} models")

        def finish():
            self.output(
                "NAS Completed\nBest Architecture:\nConv → Conv → Residual → Dense"
            )

        self.run_steps(steps, finish)

    # ---------------- COMPARISON ----------------

    def compare_generations(self, g1, g2):

        self.log(f"Comparing generation {g1} and {g2}")

        steps = [
            "Loading architectures...",
            "Evaluating metrics...",
            "Running benchmarks..."
        ]

        def finish():
            self.output(
                f"Generation {g2} performed better than generation {g1}"
            )

        self.run_steps(steps, finish)

    # ---------------- STEP ENGINE ----------------

    def run_steps(self, steps, finish_callback=None):

        def step():

            if steps:
                self.log(steps.pop(0))
                QTimer.singleShot(1000, step)
            else:
                if finish_callback:
                    finish_callback()

        step()

    # ---------------- HELPERS ----------------

    def log(self, text):

        self.training_log.append(text)

    def output(self, text):

        self.output_panel.append(text)


app = QApplication(sys.argv)

window = NASApp()
window.show()

sys.exit(app.exec())