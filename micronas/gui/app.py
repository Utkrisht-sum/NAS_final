import sys
import threading
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QTextEdit,
    QSplitter, QProgressBar, QGroupBox, QFormLayout, QFileDialog
)
from PySide6.QtCore import Qt, Signal, QObject

from gui.charts import ChartsPanel
from engine.dataset import DatasetAnalyzer
from engine.nas import NASEngine
from engine.trainer import Trainer
from engine.export import ProjectExporter
from utils.logger import get_logger

logger = get_logger("GUI")

class WorkerSignals(QObject):
    log_msg = Signal(str)
    ai_msg = Signal(str)
    nas_progress = Signal(int, float, list)  # gen, fitness, pop_data
    train_progress = Signal(int, float, float, float) # epoch, t_loss, v_loss, v_acc
    finished = Signal()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚡ MICRONAS ENGINE - AutoML")
        self.resize(1200, 800)

        self.signals = WorkerSignals()
        self.signals.log_msg.connect(self.append_log)
        self.signals.ai_msg.connect(self.set_ai_thinking)
        self.signals.nas_progress.connect(self.update_nas_chart)
        self.signals.train_progress.connect(self.update_train_chart)
        self.signals.finished.connect(self.on_finished)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # TOP PANEL: Inputs
        top_group = QGroupBox("1. Task Definition")
        top_layout = QFormLayout()

        self.dataset_input = QComboBox()
        self.dataset_input.addItems(["MNIST", "CIFAR10", "mock.csv", "Choose File/Folder..."])
        self.dataset_input.currentTextChanged.connect(self.handle_dataset_selection)
        top_layout.addRow("Dataset:", self.dataset_input)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("e.g. Train an efficient digit classifier")
        top_layout.addRow("Prompt:", self.prompt_input)

        top_group.setLayout(top_layout)
        main_layout.addWidget(top_group)

        # MIDDLE PANEL: NAS Config
        mid_group = QGroupBox("2. NAS Configuration")
        mid_layout = QHBoxLayout()

        self.pop_spin = QSpinBox()
        self.pop_spin.setRange(2, 50)
        self.pop_spin.setValue(5)

        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(1, 20)
        self.gen_spin.setValue(2)

        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 100)
        self.epoch_spin.setValue(2)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Fast Mode", "Balanced Mode", "Research Mode"])

        mid_layout.addWidget(QLabel("Population:"))
        mid_layout.addWidget(self.pop_spin)
        mid_layout.addWidget(QLabel("Generations:"))
        mid_layout.addWidget(self.gen_spin)
        mid_layout.addWidget(QLabel("Epochs:"))
        mid_layout.addWidget(self.epoch_spin)
        mid_layout.addWidget(QLabel("Mode:"))
        mid_layout.addWidget(self.mode_combo)

        mid_group.setLayout(mid_layout)
        main_layout.addWidget(mid_group)

        # DEMO PRESETS
        demo_group = QGroupBox("Quick Demos")
        demo_layout = QHBoxLayout()
        btn_demo1 = QPushButton("Demo 1: MNIST Efficient")
        btn_demo1.clicked.connect(lambda: self.load_demo("MNIST", "Train an efficient digit classifier", 5, 2))
        btn_demo2 = QPushButton("Demo 2: Fast Text/CSV")
        btn_demo2.clicked.connect(lambda: self.load_demo("mock.csv", "Build a fast model", 4, 1))

        demo_layout.addWidget(btn_demo1)
        demo_layout.addWidget(btn_demo2)
        demo_group.setLayout(demo_layout)
        main_layout.addWidget(demo_group)

        # BOTTOM PANEL: Run & Progress
        self.btn_start = QPushButton("🚀 START MICRONAS ENGINE")
        self.btn_start.setStyleSheet("font-weight: bold; font-size: 16px; padding: 10px; background-color: #2e8b57; color: white;")
        self.btn_start.clicked.connect(self.start_pipeline)
        main_layout.addWidget(self.btn_start)

        # SPLITTER: Charts & Logs
        splitter = QSplitter(Qt.Horizontal)

        # Left: Charts
        self.charts_panel = ChartsPanel()
        splitter.addWidget(self.charts_panel)

        # Right: Logs + AI Thinking
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.ai_thinking_label = QLabel("🧠 AI Thinking: Idle")
        self.ai_thinking_label.setStyleSheet("color: blue; font-weight: bold;")
        right_layout.addWidget(self.ai_thinking_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        right_layout.addWidget(self.log_output)

        splitter.addWidget(right_panel)
        main_layout.addWidget(splitter)

    def handle_dataset_selection(self, text):
        if text == "Choose File/Folder...":
            path, _ = QFileDialog.getOpenFileName(self, "Select Dataset File (CSV)", "", "CSV Files (*.csv);;All Files (*)")
            if not path:
                # If they cancel, offer folder instead
                path = QFileDialog.getExistingDirectory(self, "Select Image Dataset Folder")

            if path:
                self.dataset_input.blockSignals(True)
                self.dataset_input.insertItem(0, path)
                self.dataset_input.setCurrentIndex(0)
                self.dataset_input.blockSignals(False)
                self.append_log(f"Selected custom dataset: {path}")
            else:
                self.dataset_input.setCurrentIndex(0) # Revert to MNIST

    def load_demo(self, dataset, prompt, pop, gens):
        self.dataset_input.setCurrentText(dataset)
        self.prompt_input.setText(prompt)
        self.pop_spin.setValue(pop)
        self.gen_spin.setValue(gens)
        self.append_log(f"Loaded demo: {dataset} | {prompt}")

    def append_log(self, text):
        self.log_output.append(text)
        # auto scroll
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def set_ai_thinking(self, text):
        self.ai_thinking_label.setText(f"🧠 AI Thinking: {text}")

    def update_nas_chart(self, gen, best_fitness, pop_data):
        self.charts_panel.update_nas_chart(gen, best_fitness)
        self.charts_panel.update_pareto_chart(pop_data)

    def update_train_chart(self, epoch, t_loss, v_loss, v_acc):
        self.charts_panel.update_train_chart(epoch, t_loss, v_loss)

    def start_pipeline(self):
        self.btn_start.setEnabled(False)
        self.log_output.clear()

        dataset = self.dataset_input.currentText()
        prompt = self.prompt_input.text()
        pop = self.pop_spin.value()
        gens = self.gen_spin.value()
        epochs = self.epoch_spin.value()

        # Need a mock CSV for demo if chosen
        if dataset == "mock.csv":
            import pandas as pd
            import numpy as np
            import os
            if not os.path.exists("mock.csv"):
                df = pd.DataFrame(np.random.rand(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'target'])
                df['target'] = np.random.choice([0, 1], 100)
                df.to_csv("mock.csv", index=False)

        threading.Thread(target=self.run_pipeline_thread, args=(dataset, prompt, pop, gens, epochs), daemon=True).start()

    def run_pipeline_thread(self, dataset, prompt, pop, gens, epochs):
        try:
            self.signals.ai_msg.emit("Analyzing Dataset...")
            self.signals.log_msg.emit(f"Loading dataset: {dataset}")
            analyzer = DatasetAnalyzer(dataset, "auto")
            metadata, train_loader, val_loader = analyzer.analyze_and_load(batch_size=32)
            self.signals.log_msg.emit(f"Metadata detected: {metadata}")

            self.signals.ai_msg.emit("Parsing Prompt and Initializing NAS...")
            nas = NASEngine(metadata, train_loader, val_loader, prompt=prompt)

            # Hook the NAS output to GUI using monkey patching for the hackathon
            original_eval = nas._evaluate_fitness
            def eval_hook(model, config, device):
                res = original_eval(model, config, device)
                # Just show one evaluation message as AI thinking
                self.signals.ai_msg.emit(f"Evaluating arch with params: {res['params']} | Mem: {res['memory_mb']:.2f}MB")
                return res
            nas._evaluate_fitness = eval_hook

            self.signals.log_msg.emit(f"Running NAS. Pop: {pop}, Gens: {gens}")
            best_model, best_config, final_pop = nas.run_search(population_size=pop, generations=gens)

            # Emitting charts updates manually to simplify thread passing for hackathon
            for i in range(gens):
                # Just simulating generations chart updates
                self.signals.nas_progress.emit(i+1, nas.history[i][0] if len(nas.history) > i else 0, final_pop)

            self.signals.log_msg.emit(f"Best Config Found: {best_config}")
            self.signals.ai_msg.emit(f"NAS Completed. Best Fitness: {nas.best_model}")

            self.signals.ai_msg.emit("Starting Full Training Phase...")
            self.signals.log_msg.emit("Initializing Trainer")
            trainer = Trainer(best_model, train_loader, val_loader, task=metadata["task"])

            def train_cb(epoch, t_loss, v_loss, v_acc):
                self.signals.train_progress.emit(epoch, t_loss, v_loss, v_acc)
                self.signals.log_msg.emit(f"Epoch {epoch} | TL: {t_loss:.3f} | VL: {v_loss:.3f} | Metric: {v_acc:.2f}")

            history = trainer.train(epochs=epochs, callback=train_cb)

            self.signals.ai_msg.emit("Exporting Project...")
            exporter = ProjectExporter(best_model, metadata, history)
            exporter.export()
            self.signals.log_msg.emit("Project exported successfully to 'project_output/'")

            self.signals.ai_msg.emit("Pipeline Completed Successfully!")
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            self.signals.log_msg.emit(f"ERROR: {str(e)}\n{err}")
            self.signals.ai_msg.emit("Pipeline Failed!")
        finally:
            self.signals.finished.emit()

    def on_finished(self):
        self.btn_start.setEnabled(True)

def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
