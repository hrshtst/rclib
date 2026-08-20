#!/bin/bash

# Define input and output filenames
INPUT_MD="RLS_Optimization_Report.md"
OUTPUT_TEX="RLS_Optimization_Report.tex"
OUTPUT_PDF="RLS_Optimization_Report.pdf"
HEADER_TEX="listings_setup.tex"

# Navigate to the docs directory if the script is run from project root
if [ -d "docs" ]; then
    cd docs || exit
fi

# Check if input file exists
if [ ! -f "$INPUT_MD" ]; then
    echo "Error: Input file $INPUT_MD not found!"
    exit 1
fi

# Create a temporary header file for custom listings setup
cat <<EOF > "$HEADER_TEX"
\usepackage{xcolor}
\lstset{
    basicstyle=\ttfamily\footnotesize,
    breaklines=true,
    breakatwhitespace=false,
    postbreak=\mbox{\textcolor{red}{$\\hookrightarrow$}\space},
    frame=single,
    numbers=left,
    numberstyle=\tiny,
    showstringspaces=false,
    keywordstyle=\color{blue},
    commentstyle=\color{gray},
    stringstyle=\color{green!50!black},
    columns=fullflexible,
    keepspaces=true
}
EOF

echo "Converting $INPUT_MD to $OUTPUT_TEX using pandoc..."
# Convert MD to TeX using pandoc
# -s: standalone (creates a full latex file with preamble)
# --listings: use listings package for code blocks
# -H: include the custom header file
if ! pandoc "$INPUT_MD" -s --listings -H "$HEADER_TEX" -o "$OUTPUT_TEX"; then
    echo "Error: Pandoc conversion failed."
    rm "$HEADER_TEX"
    exit 1
fi

echo "Generating PDF from $OUTPUT_TEX using pdflatex..."
# Generate PDF using pdflatex
# Run twice to ensure proper layout
pdflatex -interaction=nonstopmode "$OUTPUT_TEX" > /dev/null
pdflatex -interaction=nonstopmode "$OUTPUT_TEX" > /dev/null

# Clean up temporary header
rm "$HEADER_TEX"

if [ -f "$OUTPUT_PDF" ]; then
    echo "Success! PDF generated at: $(pwd)/$OUTPUT_PDF"
    echo "Intermediate TeX file at: $(pwd)/$OUTPUT_TEX"
else
    echo "Error: PDF generation failed."
    exit 1
fi
