import time
from contextlib import contextmanager
from rich.console import Console
from rich.table import Table
console = Console()
@contextmanager
def stopwatch(label: str):
    s=time.time(); yield; e=time.time(); console.print(f"[bold green]{label}[/bold green] took [bold]{e-s:.2f}s[/bold]")
def print_sources(docs, scores=None, max_rows=6):
    t=Table(title="Sources"); t.add_column("Rank"); t.add_column("Page"); t.add_column("Preview"); 
    if scores is not None: t.add_column("Score")
    for i,d in enumerate(docs[:max_rows],1):
        page=d.metadata.get("page","?"); prev=(d.page_content[:180]+"...") if len(d.page_content)>200 else d.page_content
        if scores is not None: t.add_row(str(i), str(page), prev, f"{scores[i-1]:.4f}")
        else: t.add_row(str(i), str(page), prev)
    console.print(t)
