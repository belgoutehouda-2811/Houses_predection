import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from PyQt5.QtGui import QFont, QIcon, QPixmap, QMovie, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtWidgets import QGraphicsDropShadowEffect

# ===== Import model =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from training_model1 import predict_price, get_price_cluster

HISTORY_FILE = "history.csv"

# ===== Chatbot (improved + more topics) =====
def chatbot_response(text):
    text = text.lower().strip()

    # Greetings
    if any(w in text for w in ["bonjour", "salut", "salam", "hi", "hello", "مرحبا", "أهلا", "سلام"]):
        return "👋 مرحبا بك! كيفاش نقدر نعاونك اليوم؟"

    # App info
    elif any(w in text for w in ["app", "application", "تطبيق", "واش", "شنو"]):
        return "🏠 هاد التطبيق كيتنبأ بثمن الدار باستخدام 3 نماذج ديال الذكاء الاصطناعي."

    # How to use
    elif any(w in text for w in ["comment", "كيفاش", "كيف", "how", "utiliser", "استخدام"]):
        return "📝 1. عمر الخانات\n2. اختار الموديل\n3. كليك على 'Generate'"

    # Models
    elif any(w in text for w in ["model", "modèle", "نموذج", "موديل", "model"]):
        return "🧠 عندنا 3 موديلات:\n• Random Forest (الأحسن)\n• XGBoost (سريع)\n• Linear Regression (بسيط)"

    # Price related
    elif any(w in text for w in ["prix", "price", "ثمن", "سعر", "تمن"]):
        return "💰 الثمن كيتحسب حسب المساحة، عدد الغرف، والتجهيزات. كل موديل كيعطي نتيجة مختلفة شوية."

    # Area
    elif any(w in text for w in ["area", "surface", "مساحة"]):
        return "📐 خل المساحة بالمتر المربع (مثلا: 3000)."

    # Bedrooms
    elif any(w in text for w in ["bedroom", "chambre", "غرفة", "غرف"]):
        return "🛏️ دخل عدد الغرف ديال النوم (مثلا: 3)."

    # Cluster
    elif any(w in text for w in ["cluster", "classe", "فئة", "نوع"]):
        return "🏷️ كل دار كتصنف لـ 3 فئات: Budget (رخيصة), Standard (عادية), Luxury (فاخرة)."

    # History
    elif any(w in text for w in ["history", "historique", "سجل", "تاريخ"]):
        return "📊 كليك على 'Show History' باش تشوف التوقعات السابقة فنافذة جديدة."

    # Graph
    elif any(w in text for w in ["graph", "graphique", "رسم", "chart"]):
        return "📈 كليك على 'Show Graph' باش تشوف الرسم البياني ديال التوقعات."

    # Theme
    elif any(w in text for w in ["theme", "dark", "light", "وضع", "لون"]):
        return "🎨 كليك على 'Toggle Theme' باش تبدل بين الوضع الليلي والنهاري."

    # Compare
    elif any(w in text for w in ["compare", "comparer", "قارن", "مقارنة"]):
        return "⚖️ اختار 'Compare All Models' باش تشوف فرق الثمن بين 3 الموديلات."

    # Thanks
    elif any(w in text for w in ["merci", "شكرا", "thanks", "thank"]):
        return "😊 بلا جميل! أنا هنا دايما إلى احتجت شي حاجة."

    # Goodbye
    elif any(w in text for w in ["bye", "au revoir", "بسلامة", "مع السلامة"]):
        return "👋 بسلامة! ترجع فأي وقت."

    # Default
    else:
        return "🤖 ما فهمتش مزيان. جرب تسأل عن: الثمن، المساحة، الغرف، الموديلات، التاريخ، أو الرسم البياني."


# ===== Worker =====
class PredictWorker(QThread):
    result_ready = pyqtSignal(float, str)

    def __init__(self, params, model_name):
        super().__init__()
        self.params = params
        self.model_name = model_name

    def run(self):
        try:
            price = predict_price(**self.params, model_name=self.model_name)
            self.result_ready.emit(price, self.model_name)
        except Exception as e:
            self.result_ready.emit(-1, str(e))


class CompareWorker(QThread):
    results_ready = pyqtSignal(dict)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        results = {}
        for mdl in ['random_forest', 'linear_regression', 'xgboost']:
            try:
                price = predict_price(**self.params, model_name=mdl)
                results[mdl] = price
            except Exception as e:
                results[mdl] = -1
        self.results_ready.emit(results)


# ===== History Window =====
class HistoryWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle("📊 سجل التوقعات")
        self.setWindowIcon(QIcon("logo1.png"))
        self.setGeometry(600, 100, 500, 600)
        self.dark_mode = parent.dark_mode if parent else True

        self.setup_ui()
        self.load_data()
        self.apply_theme()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()

        # Title
        title = QtWidgets.QLabel("📊 Prédictions historiques")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Stats
        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setFont(QFont("Arial", 11))
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("padding: 10px; border-radius: 8px; background-color: rgba(58,134,255,0.1);")
        layout.addWidget(self.stats_label)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["#", "prix (MAD)", "historique"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        self.btn_refresh = QtWidgets.QPushButton("🔄 Actualiser ")
        self.btn_refresh.clicked.connect(self.load_data)

        self.btn_clear = QtWidgets.QPushButton("🗑️ supprimer")
        self.btn_clear.clicked.connect(self.clear_history)

        self.btn_export = QtWidgets.QPushButton("📥 enregister CSV")
        self.btn_export.clicked.connect(self.export_csv)

        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_export)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def load_data(self):
        if not os.path.exists(HISTORY_FILE):
            self.stats_label.setText("⚠️ ne pas de prediction maintenant")
            self.table.setRowCount(0)
            return

        df = pd.read_csv(HISTORY_FILE)
        if len(df) == 0:
            self.stats_label.setText("⚠️ ne pas de prediction maintenant")
            return

        # Stats
        avg = df['price'].mean()
        mx = df['price'].max()
        mn = df['price'].min()
        count = len(df)

        self.stats_label.setText(
            f"📈 la somme: {count} | 🎯 moyenne: {avg:,.0f} MAD | ⬆️ superieur: {mx:,.0f} | le bas: {mn:,.0f}"
        )

        # Table
        self.table.setRowCount(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i+1)))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{row['price']:,.0f}"))
            date = row.get('date', 'ne pas disponible')
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(date)))

        self.table.resizeColumnsToContents()

    def clear_history(self):
        reply = QtWidgets.QMessageBox.question(
            self, "sure!", 
            "tu veux supprimer le sauvgardage ",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            self.load_data()

    def export_csv(self):
        if not os.path.exists(HISTORY_FILE):
            QtWidgets.QMessageBox.warning(self, "un fautte", "ne pas de fichier")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "تصدير CSV", "predictions.csv", "CSV Files (*.csv)"
        )
        if path:
            df = pd.read_csv(HISTORY_FILE)
            df.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(self, "تم", f"تم التصدير إلى:\n{path}")

    def apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QWidget { background-color: #0f172a; color: white; font-size: 13px; }
                QTableWidget { background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; }
                QHeaderView::section { background-color: #334155; color: white; padding: 8px; border: none; }
                QPushButton { background-color: #3A86FF; color: white; border-radius: 6px; padding: 8px 16px; }
                QPushButton:hover { background-color: #2563EB; }
            """)
        else:
            self.setStyleSheet("""
                QWidget { background-color: #ffffff; color: black; font-size: 13px; }
                QTableWidget { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; }
                QHeaderView::section { background-color: #e2e8f0; color: black; padding: 8px; border: none; }
                QPushButton { background-color: #3A86FF; color: white; border-radius: 6px; padding: 8px 16px; }
                QPushButton:hover { background-color: #2563EB; }
            """)


# ===== Main App =====
class HouseApp(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("🏠 House Price Generator Pro")
        self.setWindowIcon(QIcon("logo1.png"))
        self.setGeometry(470, 30, 450, 750)

        self.dark_mode = True
        self.history_window = None

        self.setup_ui()
        self.apply_theme()
        self.apply_extra_styles()

    def setup_ui(self):
        # ===== Logo =====
        logo = QtWidgets.QLabel()
        pixmap = QPixmap("logo.png")
        logo.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo.setAlignment(Qt.AlignCenter)

        title = QtWidgets.QLabel("🏠 House Price Generator")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 20, QFont.Bold))

        subtitle = QtWidgets.QLabel("prediction avec intelligence artificiel")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setStyleSheet("color: #94a3b8;")

        # ===== Chatbot UI =====
        chat_frame = QtWidgets.QFrame()
        chat_frame.setStyleSheet("background-color: rgba(58,134,255,0.05); border-radius: 12px; padding: 8px;")
        chat_layout = QtWidgets.QVBoxLayout(chat_frame)
        chat_layout.setContentsMargins(8, 8, 8, 8)

        self.chat_display = QtWidgets.QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFixedHeight(100)
        self.chat_display.setStyleSheet("background: transparent; border: none;")
        self.chat_display.append("🤖 Bonjour, avez-vous une question ?")

        chat_input_layout = QtWidgets.QHBoxLayout()
        self.chat_input = QtWidgets.QLineEdit()
        self.chat_input.setPlaceholderText("poser votre question ...")
        self.chat_input.returnPressed.connect(self.handle_chat)

        self.chat_btn = QtWidgets.QPushButton("📤")
        self.chat_btn.setFixedSize(36, 36)
        self.chat_btn.setToolTip("إرسال")
        self.chat_btn.clicked.connect(self.handle_chat)

        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(self.chat_btn)

        chat_layout.addWidget(self.chat_display)
        chat_layout.addLayout(chat_input_layout)

        # ===== Model Selection =====
        model_frame = QtWidgets.QFrame()
        model_frame.setStyleSheet("background-color: rgba(58,134,255,0.05); border-radius: 10px; padding: 10px;")
        model_layout = QtWidgets.QVBoxLayout(model_frame)

        model_label = QtWidgets.QLabel("🧠 choisir votre model: ")
        model_label.setFont(QFont("Arial", 12, QFont.Bold))

        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.addItems([
            "🌲 model 1",
            "📈 model 2", 
            "📉 model 3 "
        ])
        self.model_selector.setStyleSheet("padding: 6px; border-radius: 6px;")

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_selector)

        # ===== Inputs =====
        input_frame = QtWidgets.QFrame()
        input_frame.setStyleSheet("background-color: rgba(255,255,255,0.02); border-radius: 10px; padding: 10px;")
        form = QtWidgets.QFormLayout(input_frame)
        form.setSpacing(10)

        self.area = QtWidgets.QLineEdit()
        self.area.setPlaceholderText("exemple: 3000")

        self.bedrooms = QtWidgets.QLineEdit()
        self.bedrooms.setPlaceholderText("exemple: 3")

        self.bathrooms = QtWidgets.QLineEdit()
        self.bathrooms.setPlaceholderText("exemple: 2")

        self.stories = QtWidgets.QLineEdit()
        self.stories.setPlaceholderText("exemple: 1")

        self.mainroad = QtWidgets.QComboBox(); self.mainroad.addItems(["yes", "no"])
        self.guestroom = QtWidgets.QComboBox(); self.guestroom.addItems(["yes", "no"])
        self.basement = QtWidgets.QComboBox(); self.basement.addItems(["yes", "no"])
        self.hotwaterheating = QtWidgets.QComboBox(); self.hotwaterheating.addItems(["yes", "no"])
        self.airconditioning = QtWidgets.QComboBox(); self.airconditioning.addItems(["yes", "no"])

        self.parkings = QtWidgets.QLineEdit()
        self.parkings.setPlaceholderText("مثلا: 1")

        self.prefarea = QtWidgets.QComboBox(); self.prefarea.addItems(["yes", "no"])

        self.city = QtWidgets.QComboBox()
        self.city.addItems(["furnished", "semi-furnished", "unfurnished"])

        # Add rows with icons
        form.addRow("📐 Surface (m²):", self.area)
        form.addRow("🛏️ Chambres à coucher:", self.bedrooms)
        form.addRow("🚿 Salles de bains:", self.bathrooms)
        form.addRow("🏢 Étages:", self.stories)
        form.addRow("🛋️ Aménagement:", self.city)
        form.addRow("🛣️ Route principale:", self.mainroad)
        form.addRow("🏠 Chambre d’invités:", self.guestroom)
        form.addRow("🏗️ Cave:", self.basement)
        form.addRow("🔥 Chauffe-eau:", self.hotwaterheating)
        form.addRow("❄️ Climatiseur:", self.airconditioning)
        form.addRow("🅿️ Parking:", self.parkings)
        form.addRow("⭐ Zone préférée", self.prefarea)

        # ===== Spinner =====
        self.loading = QtWidgets.QLabel()
        self.movie = QMovie("Spinner1234.gif")
        self.loading.setMovie(self.movie)
        self.loading.setAlignment(Qt.AlignCenter)
        self.loading.hide()

        # ===== Result =====
        self.result_frame = QtWidgets.QFrame()
        self.result_frame.setStyleSheet("""
            background-color: rgba(58,134,255,0.1); 
            border-radius: 12px; 
            border: 1px solid rgba(58,134,255,0.3);
            padding: 15px;
        """)
        result_layout = QtWidgets.QVBoxLayout(self.result_frame)

        self.result_label = QtWidgets.QLabel("💡 عمر الخانات واضغط Generate")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 14, QFont.Bold))

        self.cluster_label = QtWidgets.QLabel("")
        self.cluster_label.setAlignment(Qt.AlignCenter)
        self.cluster_label.setFont(QFont("Arial", 11))

        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.cluster_label)
        self.result_frame.hide()

        # ===== Buttons =====
        btn_layout = QtWidgets.QVBoxLayout()
        btn_layout.setSpacing(8)

        self.btn_generate = self.create_styled_button("⚡ احسب الثمن", "#3A86FF")
        self.btn_generate.clicked.connect(self.generate_price)

        self.btn_compare = self.create_styled_button("⚖️ قارن الموديلات", "#8B5CF6")
        self.btn_compare.clicked.connect(self.compare_models)

        self.btn_history = self.create_styled_button("📊 السجل", "#10B981")
        self.btn_history.clicked.connect(self.show_history)

        self.btn_graph = self.create_styled_button("📈 الرسم البياني", "#F59E0B")
        self.btn_graph.clicked.connect(self.show_graph)

        self.btn_theme = self.create_styled_button("🌙 تغيير الوضع", "#64748B")
        self.btn_theme.clicked.connect(self.toggle_theme)

        self.btn_exit = self.create_styled_button("❌ خروج", "#EF4444")
        self.btn_exit.clicked.connect(self.close)

        btn_layout.addWidget(self.btn_generate)
        btn_layout.addWidget(self.btn_compare)
        btn_layout.addWidget(self.btn_history)
        btn_layout.addWidget(self.btn_graph)
        btn_layout.addWidget(self.btn_theme)
        btn_layout.addWidget(self.btn_exit)

        # Language switch
        self.lang_selector = QtWidgets.QComboBox()
        self.lang_selector.addItems(["🇲🇦 AR", "🇬🇧 EN", "🇫🇷 FR"])
        self.lang_selector.currentIndexChanged.connect(self.change_language)
        self.lang_selector.setStyleSheet("padding: 6px; border-radius: 6px;")

        # ===== Main Layout =====
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(12)
        layout.addWidget(logo)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(chat_frame)
        layout.addWidget(model_frame)
        layout.addWidget(input_frame)
        layout.addWidget(self.btn_generate)
        layout.addWidget(self.loading)
        layout.addWidget(self.result_frame)
        layout.addWidget(self.btn_compare)
        layout.addWidget(self.btn_history)
        layout.addWidget(self.btn_graph)
        layout.addWidget(self.btn_theme)
        layout.addWidget(self.btn_exit)
        layout.addWidget(self.lang_selector)
        layout.addStretch()

        # ===== Scroll Area =====
        scroll_widget = QtWidgets.QWidget()
        scroll_widget.setLayout(layout)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_area.setStyleSheet("background: transparent;")

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

    def create_styled_button(self, text, color):
        btn = QtWidgets.QPushButton(text)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
            }}
            QPushButton:pressed {{
                background-color: {color}bb;
            }}
        """)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    # ===== Chat =====
    def handle_chat(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        self.chat_display.append(f"🧑 {text}")
        response = chatbot_response(text)
        self.chat_display.append(f"🤖 {response}")
        self.chat_input.clear()
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    # ===== Helpers =====
    def _yes_no(self, combo): return 1 if combo.currentText() == "yes" else 0
    def _furnishing_int(self): return {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}[self.city.currentText()]
    def _get_model_name(self):
        idx = self.model_selector.currentIndex()
        return ['random_forest', 'xgboost', 'linear_regression'][idx]

    # ===== Generate =====
    def generate_price(self):
        try:
            area = int(self.area.text())
            bedrooms = int(self.bedrooms.text())
            bathrooms = int(self.bathrooms.text())
            stories = int(self.stories.text())
            parking = int(self.parkings.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Erreur", "⚠️ Veuillez remplir tous les champs avec des nombres valides !")
            return

        params = dict(
            area=area, bedrooms=bedrooms, bathrooms=bathrooms, stories=stories,
            mainroad=self._yes_no(self.mainroad),
            guestroom=self._yes_no(self.guestroom),
            basement=self._yes_no(self.basement),
            hotwaterheating=self._yes_no(self.hotwaterheating),
            airconditioning=self._yes_no(self.airconditioning),
            parking=parking,
            prefarea=self._yes_no(self.prefarea),
            furnishingstatus=self._furnishing_int(),
        )

        self.loading.show()
        self.movie.start()
        self.result_frame.hide()

        self.worker = PredictWorker(params, self._get_model_name())
        self.worker.result_ready.connect(self.show_result)
        self.worker.start()

    def show_result(self, price, model_name):
        self.movie.stop()
        self.loading.hide()

        if price < 0:
            QtWidgets.QMessageBox.critical(self, "errur", f"errur: {model_name}")
            return

        model_emoji = {"random_forest": "🌲", "xgboost": "📈", "linear_regression": "📉"}
        model_display = {"random_forest": "Random Forest", "xgboost": "XGBoost", "linear_regression": "Linear Regression"}

        self.result_label.setText(f"{model_emoji.get(model_name, '🤖')} {model_display.get(model_name, model_name)}\n🏠 {price:,.0f} MAD")
        self.result_frame.show()

        # Get cluster
        try:
            cluster = get_price_cluster(
                area=int(self.area.text()),
                bedrooms=int(self.bedrooms.text()),
                bathrooms=int(self.bathrooms.text()),
                stories=int(self.stories.text()),
                mainroad=self._yes_no(self.mainroad),
                guestroom=self._yes_no(self.guestroom),
                basement=self._yes_no(self.basement),
                hotwaterheating=self._yes_no(self.hotwaterheating),
                airconditioning=self._yes_no(self.airconditioning),
                parking=int(self.parkings.text()),
                prefarea=self._yes_no(self.prefarea),
                furnishingstatus=self._furnishing_int()
            )
            self.cluster_label.setText(f"🏷️ الفئة: {cluster['label']} ({cluster['min']:,.0f} - {cluster['max']:,.0f} MAD)")
            self.cluster_label.setStyleSheet(f"color: {'#10B981' if cluster['label']=='Budget' else '#F59E0B' if cluster['label']=='Standard' else '#EF4444'};")
        except:
            self.cluster_label.setText("")

        # Save to history
        from datetime import datetime
        df = pd.DataFrame([[price, model_name, datetime.now().strftime("%Y-%m-%d %H:%M")]], 
                         columns=["price", "model", "date"])
        df.to_csv(HISTORY_FILE, mode="a", header=not os.path.exists(HISTORY_FILE), index=False)

    # ===== Compare Models =====
    def compare_models(self):
        try:
            area = int(self.area.text())
            bedrooms = int(self.bedrooms.text())
            bathrooms = int(self.bathrooms.text())
            stories = int(self.stories.text())
            parking = int(self.parkings.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Erreur", "⚠️ Veuillez remplir tous les champs avec des nombres valides !")
            return

        params = dict(
            area=area, bedrooms=bedrooms, bathrooms=bathrooms, stories=stories,
            mainroad=self._yes_no(self.mainroad),
            guestroom=self._yes_no(self.guestroom),
            basement=self._yes_no(self.basement),
            hotwaterheating=self._yes_no(self.hotwaterheating),
            airconditioning=self._yes_no(self.airconditioning),
            parking=parking,
            prefarea=self._yes_no(self.prefarea),
            furnishingstatus=self._furnishing_int(),
        )

        self.loading.show()
        self.movie.start()
        self.result_frame.hide()

        self.compare_worker = CompareWorker(params)
        self.compare_worker.results_ready.connect(self.show_comparison)
        self.compare_worker.start()

    def show_comparison(self, results):
        self.movie.stop()
        self.loading.hide()

        msg = "⚖️ comparaison entre les modeles:\n\n"
        emojis = {"random_forest": "🌲", "xgboost": "📈", "linear_regression": "📉"}
        names = {"random_forest": "Random Forest", "xgboost": "XGBoost", "linear_regression": "Linear Regression"}

        for mdl, price in results.items():
            if price > 0:
                msg += f"{emojis[mdl]} {names[mdl]}: {price:,.0f} MAD\n"
            else:
                msg += f"{emojis[mdl]} {names[mdl]}: خطأ\n"

        # Find best (middle value)
        valid = {k: v for k, v in results.items() if v > 0}
        if valid:
            best = min(valid.items(), key=lambda x: abs(x[1] - sum(valid.values())/len(valid)))[0]
            msg += f"\n✅ الأفضل: {names[best]}"

        self.result_label.setText(msg)
        self.cluster_label.setText("")
        self.result_frame.show()

    # ===== History =====
    def show_history(self):
        if self.history_window is None or not self.history_window.isVisible():
            self.history_window = HistoryWindow(self)
        self.history_window.show()
        self.history_window.raise_()
        self.history_window.activateWindow()

    # ===== Graph =====
    def show_graph(self):
        if not os.path.exists(HISTORY_FILE):
            QtWidgets.QMessageBox.information(self, "رسم بياني", "⚠️ ne pas de donnes ")
            return
        df = pd.read_csv(HISTORY_FILE)
        if len(df) == 0:
            QtWidgets.QMessageBox.information(self, "رسم بياني", "⚠️ ne pas de donnes")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Line chart
        ax1.plot(df["price"], marker="o", color="#3A86FF", linewidth=2, markersize=6)
        ax1.fill_between(range(len(df)), df["price"], alpha=0.2, color="#3A86FF")
        ax1.set_title("Historique des prix", fontsize=14, fontweight='bold')
        ax1.set_xlabel("N° de prédiction")
        ax1.set_ylabel("Prix (MAD)")
        ax1.grid(True, alpha=0.3)

        # Model distribution
        if 'model' in df.columns:
            model_counts = df['model'].value_counts()
            colors = ['#3A86FF', '#10B981', '#F59E0B']
            ax2.pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%', 
                   colors=colors[:len(model_counts)], startangle=90)
            ax2.set_title("Utilisation des modèles", fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()

    # ===== Theme =====
    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        if self.history_window and self.history_window.isVisible():
            self.history_window.dark_mode = self.dark_mode
            self.history_window.apply_theme()

    def apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QWidget {
                    background-color: #0f172a;
                    color: #e2e8f0;
                    font-size: 13px;
                }
                QLineEdit, QComboBox, QTextEdit {
                    background-color: #1e293b;
                    border: 1px solid #334155;
                    border-radius: 8px;
                    padding: 8px;
                    color: white;
                }
                QLineEdit:focus, QComboBox:focus {
                    border: 2px solid #3A86FF;
                }
                QComboBox::drop-down {
                    border: none;
                    padding-right: 10px;
                }
                QComboBox QAbstractItemView {
                    background-color: #1e293b;
                    color: white;
                    selection-background-color: #3A86FF;
                }
            """)
            self.btn_theme.setText("☀️ تغيير الوضع")
        else:
            self.setStyleSheet("""
                QWidget {
                    background-color: #f8fafc;
                    color: #1e293b;
                    font-size: 13px;
                }
                QLineEdit, QComboBox, QTextEdit {
                    background-color: white;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    padding: 8px;
                    color: black;
                }
                QLineEdit:focus, QComboBox:focus {
                    border: 2px solid #3A86FF;
                }
                QComboBox::drop-down {
                    border: none;
                    padding-right: 10px;
                }
                QComboBox QAbstractItemView {
                    background-color: white;
                    color: black;
                    selection-background-color: #3A86FF;
                }
            """)
            self.btn_theme.setText("🌙 تغيير الوضع")

    def change_language(self):
        lang = self.lang_selector.currentText()

        if "AR" in lang:
            self.btn_generate.setText("⚡ احسب الثمن")
            self.btn_compare.setText("⚖️ قارن الموديلات")
            self.btn_history.setText("📊 السجل")
            self.btn_graph.setText("📈 الرسم البياني")
            self.btn_theme.setText("☀️ تغيير الوضع")
            self.btn_exit.setText("❌ خروج")
            self.chat_btn.setToolTip("إرسال")
            self.chat_input.setPlaceholderText("اسأل هنا...")
        elif "FR" in lang:
            self.btn_generate.setText("⚡ Générer Prix")
            self.btn_compare.setText("⚖️ Comparer Modèles")
            self.btn_history.setText("📊 Historique")
            self.btn_graph.setText("📈 Graphique")
            self.btn_theme.setText("☀️ Thème")
            self.btn_exit.setText("❌ Quitter")
            self.chat_btn.setToolTip("Envoyer")
            self.chat_input.setPlaceholderText("Posez votre question...")
        else:
            self.btn_generate.setText("⚡ Generate Price")
            self.btn_compare.setText("⚖️ Compare All Models")
            self.btn_history.setText("📊 Show History")
            self.btn_graph.setText("📈 Show Graph")
            self.btn_theme.setText("☀️ Toggle Theme")
            self.btn_exit.setText("❌ Exit")
            self.chat_btn.setToolTip("Send")
            self.chat_input.setPlaceholderText("Ask here...")

    def apply_extra_styles(self):
        self.setCursor(Qt.PointingHandCursor)

        # Shadow effect for result frame
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(58, 134, 255, 80))
        shadow.setOffset(0, 4)
        self.result_frame.setGraphicsEffect(shadow)


# ===== Run =====
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = HouseApp()
    window.show()
    sys.exit(app.exec_())