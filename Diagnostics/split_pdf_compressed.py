#!/usr/bin/env python3
"""
Script to split a large PDF into smaller parts with proper compression.
Usage: python split_pdf_compressed.py input.pdf --parts 3
"""
import argparse
import os
import subprocess
import math


def split_pdf_with_ghostscript(input_path, num_parts=3, output_dir=None):
    """
    Split a PDF using Ghostscript for better compression.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input PDF not found: {input_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path))
        if not output_dir:
            output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    # First, get the total number of pages using pdfinfo or gs
    try:
        result = subprocess.run(['pdfinfo', input_path], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Pages:'):
                    total_pages = int(line.split(':')[1].strip())
                    break
        else:
            raise Exception("pdfinfo failed")
    except:
        # Fallback: use ghostscript to count pages
        try:
            cmd = ['gs', '-q', '-dNODISPLAY', '-c', f'({input_path}) (r) file runpdfbegin pdfpagecount = quit']
            result = subprocess.run(cmd, capture_output=True, text=True)
            total_pages = int(result.stdout.strip())
        except:
            print("Error: Could not determine page count. Install poppler-utils (pdfinfo) or ghostscript.")
            return
    
    print(f"Total pages: {total_pages}")
    
    if total_pages < num_parts:
        print(f"Warning: PDF has only {total_pages} pages, cannot split into {num_parts} parts")
        num_parts = total_pages
    
    # Calculate pages per part
    pages_per_part = math.ceil(total_pages / num_parts)
    print(f"Pages per part: {pages_per_part}")
    
    # Generate base filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Split using ghostscript
    for part_num in range(num_parts):
        start_page = part_num * pages_per_part + 1
        end_page = min((part_num + 1) * pages_per_part, total_pages)
        
        if start_page > total_pages:
            break
        
        # Generate output filename
        output_filename = f"{base_name}_part{part_num + 1}_pages{start_page}-{end_page}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Creating part {part_num + 1}: {output_filename} (pages {start_page}-{end_page})")
        
        # Use ghostscript to extract pages with compression
        cmd = [
            'gs',
            '-sDEVICE=pdfwrite',
            '-dCompatibilityLevel=1.4',
            '-dPDFSETTINGS=/screen',  # Compress images
            '-dNOPAUSE',
            '-dQUIET',
            '-dBATCH',
            f'-dFirstPage={start_page}',
            f'-dLastPage={end_page}',
            f'-sOutputFile={output_path}',
            input_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error creating part {part_num + 1}: {result.stderr}")
                continue
            
            # Check file size
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"  Saved: {output_path} ({file_size:.1f} MB)")
            else:
                print(f"  Error: Output file not created")
                
        except FileNotFoundError:
            print("Error: Ghostscript (gs) not found. Please install ghostscript.")
            return
        except Exception as e:
            print(f"Error creating part {part_num + 1}: {e}")
    
    print(f"\nSplit complete! Created {num_parts} parts in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a large PDF into smaller compressed parts using Ghostscript."
    )
    parser.add_argument(
        "input_pdf",
        help="Path to the input PDF file."
    )
    parser.add_argument(
        "--parts", type=int, default=3,
        help="Number of parts to split into (default: 3)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: same as input file)."
    )
    
    args = parser.parse_args()
    
    try:
        split_pdf_with_ghostscript(args.input_pdf, args.parts, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
