"""Tkinter PDF preview compatible with the PyMuPDF used by CloudCompare 2.14."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import fitz


class ShowPdf:
    """Render a PDF incrementally in a scrollable Tkinter text widget."""

    def __init__(self) -> None:
        self.img_object_li = []
        self._document = None

    def pdf_view(
        self,
        master,
        width=1200,
        height=600,
        pdf_location="",
        bar=True,
        load="after",
    ):
        frame = tk.Frame(master, width=width, height=height, bg="white")
        scroll_y = ttk.Scrollbar(frame, orient="vertical")
        scroll_x = ttk.Scrollbar(frame, orient="horizontal")
        text = tk.Text(
            frame,
            width=width,
            height=height,
            wrap="none",
            xscrollcommand=scroll_x.set,
            yscrollcommand=scroll_y.set,
        )

        scroll_x.configure(command=text.xview)
        scroll_y.configure(command=text.yview)
        scroll_x.pack(fill="x", side="bottom")
        scroll_y.pack(fill="y", side="right")

        progress_text = tk.StringVar(value="Loading PDF…")
        progress_label = ttk.Label(frame, textvariable=progress_text)
        progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate")
        if bar:
            progress_label.pack(side="top", fill="x")
            progress.pack(side="top", fill="x")
        text.pack(side="left", fill="both", expand=True)

        def finish() -> None:
            if self._document is not None:
                self._document.close()
                self._document = None
            if bar:
                progress.pack_forget()
                progress_label.pack_forget()
            text.configure(state="disabled")

        def show_error(error: Exception) -> None:
            text.insert("end", f"Could not display the PDF:\n{error}")
            finish()

        def render_page(page_number: int) -> None:
            try:
                if self._document is None or page_number >= len(self._document):
                    finish()
                    return

                page = self._document.load_page(page_number)
                pixmap = page.get_pixmap(alpha=False)
                image = tk.PhotoImage(data=pixmap.tobytes("ppm"))
                self.img_object_li.append(image)
                text.image_create("end", image=image)
                text.insert("end", "\n\n")

                completed = page_number + 1
                total = len(self._document)
                progress["maximum"] = total
                progress["value"] = completed
                progress_text.set(f"Loading PDF… {completed}/{total}")
                master.after(1, render_page, completed)
            except Exception as error:
                show_error(error)

        def start_loading() -> None:
            try:
                self.img_object_li.clear()
                self._document = fitz.open(pdf_location)
                render_page(0)
            except Exception as error:
                show_error(error)

        if load == "after":
            master.after(250, start_loading)
        else:
            start_loading()

        return frame
