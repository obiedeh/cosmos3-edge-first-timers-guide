"""Build cosmos3_edge_first_timers_guide.ipynb from notebook_src.txt.

The notebook's cells are kept as plain text so they can be edited and reviewed
like any other source file. A line that is exactly `#%% md` or `#%% code`
starts a new cell. Edit notebook_src.txt, then run:

    python build_notebook.py [output_path]

The built notebook carries no outputs. Colab metadata asks for a GPU runtime.
"""
import re, sys, nbformat as nbf

src = open("notebook_src.txt", encoding="utf-8").read()
cells = []
for kind, body in re.findall(r"(?ms)^#%% (md|code)\n(.*?)(?=^#%% (?:md|code)\n|\Z)", src):
    body = body.strip("\n")
    cells.append(nbf.v4.new_markdown_cell(body) if kind == "md" else nbf.v4.new_code_cell(body))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
nbf.validate(nb)
out = sys.argv[1] if len(sys.argv) > 1 else "cosmos3_edge_first_timers_guide.ipynb"
nbf.write(nb, out)
print("cells:", len(cells), "->", out)
