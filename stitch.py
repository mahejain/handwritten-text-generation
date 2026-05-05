from PIL import Image
import os

GLYPH_DIR = "glyphs"
OUTPUT_DIR = "output"
GLYPH_SIZE = 64        
SPACING = 4            
PADDING = 10           

def load_glyph(char):
    filename = f"{ord(char)}.png"
    path = os.path.join(GLYPH_DIR, filename)
    
    if not os.path.exists(path):
        print(f"Warning: No glyph found for '{char}', skipping...")
        return None
    
    return Image.open(path)

def stitch_text(text, output_filename="output.png"):
    
    glyphs = []
    
    for char in text:
        if char == " ":
            blank = Image.new("L", (GLYPH_SIZE // 2, GLYPH_SIZE), color=255)
            glyphs.append(blank)
        else:
            glyph = load_glyph(char)
            if glyph:
                glyphs.append(glyph)
    
    if not glyphs:
        print("No glyphs to stitch!")
        return
    
    total_width = sum(g.width for g in glyphs) + SPACING * (len(glyphs) - 1) + PADDING * 2
    total_height = GLYPH_SIZE + PADDING * 2
    
    canvas = Image.new("L", (total_width, total_height), color=255)
    
    x = PADDING
    for glyph in glyphs:
        canvas.paste(glyph, (x, PADDING))
        x += glyph.width + SPACING
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    canvas.save(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    stitch_text("Hello World", "hello_world.png")
    stitch_text("I love Python", "i_love_python.png")
    stitch_text("mahejain", "my_name.png")