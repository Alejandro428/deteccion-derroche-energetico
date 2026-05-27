from __future__ import annotations

import sys
from pathlib import Path

import flet as ft

try:
    from .ui.dashboard import build_dashboard
except ImportError:
    # Permite ejecutar este archivo directamente sin contexto de paquete.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.app.ui.dashboard import build_dashboard


def main(page: ft.Page) -> None:
    page.title = "Prediccion de derroche energetico"
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(use_material3=True)
    page.padding = 0
    page.scroll = ft.ScrollMode.AUTO
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window_min_width = 980
    page.window_min_height = 760

    build_dashboard(page)


if __name__ == "__main__":
    ft.run(main)
