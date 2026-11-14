<div align="center">

# 🕰️ Timo 
### Your cute AI focus companion that keeps your eyes on the prize!

[![Download](https://img.shields.io/badge/Download-Timo_Installer.exe-blue?style=for-the-badge&logo=windows)](https://github.com/YOUR_USERNAME/timo/releases)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![Privacy](https://img.shields.io/badge/Privacy-100%25_Offline-success?style=for-the-badge&logo=shield)]()

</div>

---

## ✨ What is Timo?

Timo is a cozy, local-first desktop companion designed to help you study and work without burning out. He lives on your screen, watches you through your webcam (locally!), and uses AI to figure out if you're **Focusing**, **Distracted**, **Tired**, or **Away**. 

If you lose focus, Timo will gently nudge you with funny animations and encouraging words. If you've been working hard, he'll suggest a well-deserved break. 

> 🛡️ **Privacy First:** Timo processes everything directly on your machine. No internet, no cloud, no accounts, and no images are ever saved to your disk.

---

## 🌟 Features

- **🧠 Smart AI Fusion:** Timo doesn't just guess. He uses a custom-trained **YOLO** image model combined with an **XGBoost** model that reads your head pose. It's like having a tiny data scientist in your clock!
- **🎯 Auto-Calibration:** Sit in front of the camera for 15 seconds, and Timo learns your natural resting face so he doesn't falsely accuse you of slacking off.
- **🍅 Pomodoro Magic:** Built-in 25/5 focus and break timers. 
- **📝 Session To-Do List:** Add tasks for your session. When you tick them off, Timo does a happy little jump to celebrate!
- **🎨 Cute Animations:** Timo reads a book when you focus, looks grumpy when you get distracted, and meditates when it's time to rest.
- **🛑 Ctrl+C Support:** For the devs, you can safely kill the app from the terminal without it freezing.

---

## 🚀 How to Install

Getting Timo on your computer is super easy:

1. Go to the **[Releases](https://github.com/YOUR_USERNAME/timo/releases)** tab on the right side of this page.
2. Download the latest **`Timo_Installer.exe`**.
3. Run the installer (it will create a shortcut on your desktop).
4. Double-click the **Timo** icon, wait ~5 seconds for the AI brains to wake up, and start focusing!

> ⚠️ **Note for Windows Users:** Because Timo is a new, independent app, Windows SmartScreen might say "Windows protected your PC." Just click **"More info"** and then **"Run anyway"**—Timo is perfectly safe!

---

## 🛠️ The Tech Stack (For the curious)

Timo was built entirely from scratch during a 7-day sprint:
- **UI:** `PySide6` (Qt) for the beautiful, frameless transparent overlays and dashboard.
- **Vision:** `MediaPipe` for Face Landmarkers (Eye Aspect Ratio, Yawning, Head Yaw/Pitch).
- **Image AI:** `Ultralytics YOLO11` (Fine-tuned on a Kaggle Focused/Distracted dataset).
- **Logic AI:** `XGBoost` (Trained on the Mendeley Student Attention dataset).
- **Packaging:** `PyInstaller` + `Inno Setup` to create the final executable.

---

<div align="center">
  <i>Take a deep breath, put your phone away, and let's get to work. Timo is watching! 🦉</i>
</div>