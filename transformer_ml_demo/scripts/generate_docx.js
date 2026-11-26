const fs = require("fs");
const path = require("path");
const { imageSize } = require("image-size");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  HeadingLevel,
  ImageRun,
  LevelFormat,
  PageNumber,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

const root = path.resolve(__dirname, "..");
const inputPath = path.join(root, "docs", "transformer_guide.md");
const outputPath = path.join(root, "docs", "transformer_guide.docx");
const markdown = fs.readFileSync(inputPath, "utf8");
const lines = markdown.split(/\r?\n/);
const cjkFont = "PingFang SC";
const latinFont = "Arial";
const contentWidth = 9906;
const border = { style: BorderStyle.SINGLE, size: 1, color: "CBD5E1" };
const borders = {
  top: border,
  bottom: border,
  left: border,
  right: border,
};

function font(size, extra = {}) {
  return {
    size,
    font: { ascii: latinFont, hAnsi: latinFont, eastAsia: cjkFont },
    ...extra,
  };
}

function inlineRuns(text, options = {}) {
  const runs = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > cursor) {
      runs.push(new TextRun({ text: text.slice(cursor, match.index), ...options }));
    }
    const token = match[0];
    if (token.startsWith("`")) {
      runs.push(
        new TextRun({
          text: token.slice(1, -1),
          font: "Menlo",
          color: "0F766E",
          ...options,
        })
      );
    } else if (token.startsWith("**")) {
      runs.push(
        new TextRun({ text: token.slice(2, -2), bold: true, ...options })
      );
    } else {
      const link = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      runs.push(
        new TextRun({
          text: `${link[1]} (${link[2]})`,
          color: "2563EB",
          ...options,
        })
      );
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) {
    runs.push(new TextRun({ text: text.slice(cursor), ...options }));
  }
  return runs.length ? runs : [new TextRun({ text, ...options })];
}

function markdownTable(rows) {
  const cells = rows.map((row) =>
    row
      .trim()
      .replace(/^\||\|$/g, "")
      .split("|")
      .map((value) => value.trim())
  );
  const columnCount = cells[0].length;
  const baseWidth = Math.floor(contentWidth / columnCount);
  const widths = Array(columnCount).fill(baseWidth);
  widths[columnCount - 1] += contentWidth - baseWidth * columnCount;
  return new Table({
    width: { size: contentWidth, type: WidthType.DXA },
    columnWidths: widths,
    rows: cells.map(
      (row, rowIndex) =>
        new TableRow({
          cantSplit: true,
          children: row.map(
            (value, columnIndex) =>
              new TableCell({
                borders,
                width: { size: widths[columnIndex], type: WidthType.DXA },
                verticalAlign: VerticalAlign.CENTER,
                shading:
                  rowIndex === 0
                    ? { fill: "DDEBF7", type: ShadingType.CLEAR }
                    : undefined,
                margins: { top: 80, bottom: 80, left: 100, right: 100 },
                children: [
                  new Paragraph({
                    children: inlineRuns(value, {
                      bold: rowIndex === 0,
                      size: 19,
                    }),
                    spacing: { after: 0 },
                  }),
                ],
              })
          ),
        })
    ),
  });
}

function imageParagraph(relativePath, altText) {
  const imagePath = path.resolve(path.dirname(inputPath), relativePath);
  const data = fs.readFileSync(imagePath);
  const dimensions = imageSize(data);
  const maxWidth = 500;
  const maxHeight = 520;
  const scale = Math.min(
    maxWidth / dimensions.width,
    maxHeight / dimensions.height,
    1
  );
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [
      new ImageRun({
        type: "png",
        data,
        transformation: {
          width: Math.round(dimensions.width * scale),
          height: Math.round(dimensions.height * scale),
        },
        altText: { title: altText, description: altText, name: altText },
      }),
    ],
    spacing: { before: 120, after: 100 },
  });
}

const title = lines.find((line) => line.startsWith("# "))?.slice(2) || "Transformer";
const children = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 2500, after: 360 },
    children: [
      new TextRun({
        text: title,
        ...font(42, { bold: true, color: "17365D" }),
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [
      new TextRun({
        text: "Encoder、Decoder、Block 结构、逐步张量维度与 PyTorch 实战",
        ...font(25, { color: "4B5563" }),
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 800 },
    children: [
      new TextRun({
        text: "数据集：OPUS-100 en-zh  |  实现：From-Scratch PyTorch",
        ...font(20, { color: "64748B" }),
      }),
    ],
  }),
];

let inCode = false;
let codeLines = [];
let numberReferenceIndex = 0;
let previousWasNumber = false;
let firstContentHeading = true;

for (let index = 1; index < lines.length; index += 1) {
  const line = lines[index];
  if (line.startsWith("```")) {
    if (inCode) {
      for (const codeLine of codeLines) {
        children.push(
          new Paragraph({
            shading: { fill: "F1F5F9", type: ShadingType.CLEAR },
            indent: { left: 240, right: 240 },
            spacing: { before: 0, after: 0 },
            children: [
              new TextRun({
                text: codeLine || " ",
                font: "Menlo",
                size: 18,
                color: "1E293B",
              }),
            ],
          })
        );
      }
      children.push(new Paragraph({ spacing: { after: 100 } }));
      codeLines = [];
      inCode = false;
    } else {
      inCode = true;
    }
    continue;
  }
  if (inCode) {
    codeLines.push(line);
    continue;
  }

  const image = line.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
  if (image) {
    children.push(imageParagraph(image[2], image[1] || "Transformer architecture"));
    previousWasNumber = false;
    continue;
  }

  if (line.trim().startsWith("|") && index + 1 < lines.length) {
    const separator = lines[index + 1];
    if (/^\s*\|?[\s:|-]+\|/.test(separator)) {
      const tableLines = [line];
      index += 2;
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        tableLines.push(lines[index]);
        index += 1;
      }
      index -= 1;
      children.push(markdownTable(tableLines));
      children.push(new Paragraph({ spacing: { after: 100 } }));
      previousWasNumber = false;
      continue;
    }
  }

  const heading = line.match(/^(#{2,4})\s+(.+)$/);
  if (heading) {
    const level = {
      2: HeadingLevel.HEADING_1,
      3: HeadingLevel.HEADING_2,
      4: HeadingLevel.HEADING_3,
    }[heading[1].length];
    children.push(
      new Paragraph({
        heading: level,
        pageBreakBefore: firstContentHeading,
        children: inlineRuns(heading[2]),
      })
    );
    firstContentHeading = false;
    previousWasNumber = false;
    continue;
  }

  const bullet = line.match(/^\s*-\s+(.+)$/);
  if (bullet) {
    children.push(
      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: inlineRuns(bullet[1]),
        spacing: { after: 60 },
      })
    );
    previousWasNumber = false;
    continue;
  }

  const number = line.match(/^\s*\d+\.\s+(.+)$/);
  if (number) {
    if (!previousWasNumber) numberReferenceIndex += 1;
    children.push(
      new Paragraph({
        numbering: {
          reference: `numbers-${Math.min(numberReferenceIndex, 30)}`,
          level: 0,
        },
        children: inlineRuns(number[1]),
        spacing: { after: 60 },
      })
    );
    previousWasNumber = true;
    continue;
  }

  if (!line.trim()) {
    previousWasNumber = false;
    continue;
  }

  const isFormula = line.trim().startsWith("$$") || line.trim().endsWith("$$");
  children.push(
    new Paragraph({
      alignment: isFormula ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
      spacing: { after: 110, line: 340 },
      children: inlineRuns(line, {
        font: isFormula ? "Cambria Math" : undefined,
        italics: isFormula,
      }),
    })
  );
  previousWasNumber = false;
}

const numbering = [
  {
    reference: "bullets",
    levels: [
      {
        level: 0,
        format: LevelFormat.BULLET,
        text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 600, hanging: 300 } } },
      },
    ],
  },
];
for (let index = 1; index <= 30; index += 1) {
  numbering.push({
    reference: `numbers-${index}`,
    levels: [
      {
        level: 0,
        format: LevelFormat.DECIMAL,
        text: "%1.",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 600, hanging: 300 } } },
      },
    ],
  });
}

const doc = new Document({
  numbering: { config: numbering },
  styles: {
    default: {
      document: {
        run: font(21),
        paragraph: { spacing: { line: 340 } },
      },
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: font(31, { bold: true, color: "17365D" }),
        paragraph: {
          spacing: { before: 260, after: 140 },
          outlineLevel: 0,
          keepNext: false,
          keepLines: false,
        },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: font(26, { bold: true, color: "1F4E79" }),
        paragraph: {
          spacing: { before: 220, after: 100 },
          outlineLevel: 1,
          keepNext: false,
          keepLines: false,
        },
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: font(23, { bold: true, color: "365F91" }),
        paragraph: {
          spacing: { before: 180, after: 80 },
          outlineLevel: 2,
          keepNext: false,
          keepLines: false,
        },
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 900, right: 1000, bottom: 900, left: 1000 },
        },
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ text: "Transformer 中英翻译实战  |  ", ...font(17, { color: "64748B" }) }),
                new TextRun({ children: [PageNumber.CURRENT], ...font(17, { color: "64748B" }) }),
              ],
            }),
          ],
        }),
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outputPath, buffer);
  console.log(outputPath);
});
