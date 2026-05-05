from PIL import Image, ImageDraw, ImageFont
import os

# Settings
IMG_SIZE = (64, 64)        
BG_COLOR = 255             
TEXT_COLOR = 0             
FONT_SIZE = 40
SAVE_DIR = "glyphs"

def generate_glyphs():
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()
        print("Using default font — we'll upgrade this soon!")

    for char in characters:
        img = Image.new("L", IMG_SIZE, color=BG_COLOR)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (IMG_SIZE[0] - w) // 2
        y = (IMG_SIZE[1] - h) // 2

        draw.text((x, y), char, fill=TEXT_COLOR, font=font)

        filename = f"{ord(char)}.png"
        img.save(os.path.join(SAVE_DIR, filename))
        print(f"Saved: {char} → {filename}")

    print("\n✅ All glyphs generated!")

if __name__ == "__main__":
    generate_glyphs()