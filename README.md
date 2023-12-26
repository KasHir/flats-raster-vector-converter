# flats-raster-vector-converter

A Python tool for transforming bitmap images into optimized SVG paths, suitable for various NC machines, including CNC milling and laser cutters. Key features:

* Versatile Application: Ideal for CNC milling G-code generation and adaptable for use with laser cutters and other NC machines.
* Advanced Image Processing: Implements image skeletonization and other techniques for precise vector conversion.
* Graph Theory Optimization: Utilizes graph theory for efficient path planning.
* TSP Solver Integration: Incorporates OR-Tools' TSP solver for optimal routing.

Designed for professionals and DIY enthusiasts seeking efficient machining and creative applications.

![process_overview](github_docs/imgs/process_overview.png)

## Installation and Dependencies

To use the flats-raster-vector-converter, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/KasHir/flats-raster-vector-converter
```

2. Navigate to the project directory:

```bash
cd flats-raster-vector-converter
```

Ensure you have Python installed on your system. This tool has been tested on Python 3.11.6.


## Example Use

### Input
- **File Format**: PNG (raster image)
- **Example Command**: `python converter.py input.png output.svg`

The ideal input is a line-art style raster image, which can be prepared using image processing tools such as the ControlNet processor with Canny edge detection or line-art conversions.

### Output
- **File Format**: SVG (vector image)
- **Details**: The output SVG file will contain the optimized vector paths suitable for CNC milling and other NC machines.

## Limitations and Known Issues

- Currently supports only PNG format for input. Future updates aim to include more formats.
- Performance may vary with very large images or complex patterns.
- requirements.txt will be prepared in the future.
- [List any other known issues or limitations here]

## Contact

For more information, feedback, or questions, feel free to reach out:

- **Website**: [f/ats](https://f-l-ats.blogspot.com/)
- **X (Twitter)**: [@flatsCircle](https://twitter.com/flatsCircle)
- **X (Twitter)**: [@fish_meat](https://twitter.com/fish_meat)

We welcome your input and contributions to this project!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The MIT License is a permissive license that is short and to the point. It lets people do anything they want with your code as long as they provide attribution back to you and don’t hold you liable.

