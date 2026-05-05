from PIL import Image, ImageDraw, ImageFont
import random
import os

FONT_PATH = "C:/Windows/Fonts/segoepr.ttf"
OUTPUT_DIR = "output"
BG_COLOR = 255
TEXT_COLOR = 0

def render_handwritten(text, output_filename="level2_output.png", font_size=80):
    
    font = ImageFont.truetype(FONT_PATH, font_size)
    padding = 40
    spacing = 10
    
    # First pass — render each character and collect images
    letter_imgs = []
    
    for char in text:
        if char == " ":
            letter_imgs.append(None)
            continue
        
        varied_size = font_size + random.randint(-4, 4)
        varied_font = ImageFont.truetype(FONT_PATH, varied_size)
        
        # Make a big temp canvas
        temp = Image.new("L", (300, 300), color=BG_COLOR)
        draw = ImageDraw.Draw(temp)
        
        # Draw at a fixed position with lots of room
        draw.text((50, 50), char, fill=TEXT_COLOR, font=varied_font)
        
        # Get bbox AFTER drawing at known position
        bbox = draw.textbbox((50, 50), char, font=varied_font)
        
        # Crop using the actual bbox with a little padding
        pad = 8
        cropped = temp.crop((
            bbox[0] - pad,
            bbox[1] - pad,
            bbox[2] + pad,
            bbox[3] + pad
        ))
        
        # Slight rotation
        angle = random.uniform(-5, 5)
        rotated = cropped.rotate(angle, fillcolor=BG_COLOR, expand=True)
        
        letter_imgs.append(rotated)
    
    # Calculate total canvas size
    space_width = font_size // 2
    total_width = padding * 2
    max_height = 0
    
    for img in letter_imgs:
        if img is None:
            total_width += space_width
        else:
            total_width += img.width + spacing
            max_height = max(max_height, img.height)
    
    canvas_height = max_height + padding * 2 + 20
    canvas = Image.new("L", (total_width, canvas_height), color=BG_COLOR)
    
    # Paste letters onto canvas
    x = padding
    for img in letter_imgs:
        if img is None:
            x += space_width
            continue
        
        wobble = random.randint(-4, 4)
        y = padding + wobble
        canvas.paste(img, (x, y))
        x += img.width + spacing
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    canvas.save(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    render_handwritten("Hello World", "level2_hello.png")
    render_handwritten("mahejain", "level2_name.png")
    render_handwritten("I love Python", "level2_python.png")