import openslide
from monai.transforms import Compose, ScaleIntensity, ToTensor
from PIL import Image
import torch
import numpy as np
from pathlib import Path
import csv
import time

class WSIAnalyzer:
    """Whole Slide Image analyzer using MONAI framework"""
    
    def __init__(self, tile_size=256, stride=256):
        self.tile_size = tile_size
        self.stride = stride
        self.setup_transforms()
        self.setup_model()
        
    def setup_transforms(self):
        """Setup MONAI preprocessing pipeline"""
        self.preprocess = Compose([
            ScaleIntensity(),  # Normalize to [0,1]
            ToTensor()         # Convert to tensor
        ])
        
    def setup_model(self):
        """Setup demonstration AI model"""
        from monai.networks.nets import DenseNet121
        
        # Create model (normally would load pre-trained weights)
        self.model = DenseNet121(
            spatial_dims=2,
            in_channels=3,
            out_channels=2  # Binary: normal vs abnormal
        )
        self.model.eval()  # Set to evaluation mode
        
    def analyze_slide(self, slide_path, output_dir="ai_results"):
        """Analyze whole slide image with AI"""
        
        print(f"🔬 Starting AI analysis of: {slide_path}")
        
        # Load slide
        slide = openslide.OpenSlide(str(slide_path))
        width, height = slide.dimensions
        
        # Calculate analysis scope
        total_tiles = (width // self.stride) * (height // self.stride)
        print(f"📊 Slide dimensions: {width:,} x {height:,} pixels")
        print(f"🔢 Will analyze {total_tiles:,} tiles of {self.tile_size}x{self.tile_size} pixels")
        print(f"⏱️ Estimated time: {total_tiles * 0.01:.1f} seconds")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Analysis results storage
        predictions = []
        start_time = time.time()
        
        # Process tiles across the slide
        tile_count = 0
        with torch.no_grad():
            for y in range(0, height, self.stride):
                for x in range(0, width, self.stride):
                    # Extract tile
                    tile = slide.read_region((x, y), 0, (self.tile_size, self.tile_size))
                    tile = tile.convert("RGB")
                    
                    # Skip mostly background tiles
                    if self.is_mostly_background(tile):
                        continue
                    
                    # Preprocess for AI
                    tile_array = np.array(tile)  # Convert from PIL to NumPy
                    tile_tensor = self.preprocess(tile_array).unsqueeze(0)
                    
                    # AI prediction (using random values for demonstration)
                    # In real application, this would be: output = self.model(tile_tensor)
                    output = torch.randn(1, 2)  # Simulated model output
                    
                    # Convert to prediction
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = torch.softmax(output, dim=1).max().item()
                    
                    # Store result
                    predictions.append({
                        'x': x, 'y': y,
                        'prediction': prediction,
                        'confidence': confidence,
                        'tile_id': f"tile_{x}_{y}"
                    })
                    
                    tile_count += 1
                    
                    # Progress update
                    if tile_count % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = tile_count / elapsed
                        print(f"   Processed {tile_count:,} tiles ({rate:.1f} tiles/sec)")
        
        # Analysis summary
        abnormal_tiles = sum(1 for p in predictions if p['prediction'] == 1)
        total_analyzed = len(predictions)
        
        print(f"\n📈 Analysis Complete!")
        print(f"   Total tiles analyzed: {total_analyzed:,}")
        print(f"   Abnormal regions detected: {abnormal_tiles:,}")
        print(f"   Percentage abnormal: {abnormal_tiles/total_analyzed*100:.1f}%")
        print(f"   Analysis time: {time.time() - start_time:.1f} seconds")
        
        # Save results
        self.save_results(predictions, output_path)
        
        return predictions
    
    def is_mostly_background(self, tile, threshold=240):
        """Check if tile is mostly background (white space)"""
        # Convert to grayscale and check average intensity
        gray = tile.convert('L')
        avg_intensity = np.array(gray).mean()
        return avg_intensity > threshold
    
    def save_results(self, predictions, output_path):
        """Save analysis results to CSV"""
        
        results_file = output_path / "ai_predictions.csv"
        
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['x', 'y', 'prediction', 'confidence', 'tile_id'])
            writer.writeheader()
            writer.writerows(predictions)
        
        print(f"✅ Results saved to: {results_file}")
        
        # Create QuPath-compatible annotations
        self.create_qupath_annotations(predictions, output_path)
    
    def create_qupath_annotations(self, predictions, output_path, confidence_threshold=0.7):
        """Create QuPath-compatible annotation file"""
        
        # Filter high-confidence abnormal predictions
        high_conf_abnormal = [
            p for p in predictions 
            if p['prediction'] == 1 and p['confidence'] >= confidence_threshold
        ]
        
        annotations_file = output_path / "qupath_annotations.csv"
        
        with open(annotations_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Name", "Class", "Parent", "ROI", "Measurements"])
            
            for i, pred in enumerate(high_conf_abnormal):
                x, y = pred['x'], pred['y']
                confidence = pred['confidence']
                
                roi_string = f"RECTANGLE ({x} {y} {self.tile_size} {self.tile_size})"
                measurements = f"AI_Confidence:{confidence:.3f};Method:MONAI_Demo"
                
                writer.writerow([
                    f"AI_Abnormal_{i+1}",
                    "Abnormal",
                    "",
                    roi_string,
                    measurements
                ])
        
        print(f"✅ QuPath annotations saved: {annotations_file}")
        print(f"   Created {len(high_conf_abnormal)} high-confidence annotations")

# Execute analysis
def main():
    print("🚀 MONAI Whole Slide Image Analysis")
    print("====================================")
    
    # Find slide files
    slide_dir = Path("slides")
    svs_files = list(slide_dir.rglob("*.svs"))
    
    if not svs_files:
        print("❌ No .svs files found in slides/ directory")
        print("Please ensure you have downloaded slides in Task 3")
        return
    
    # Analyze first slide
    slide_path = svs_files[0]
    print(f"🔍 Analyzing: {slide_path.name}")
    
    # Create analyzer and run analysis
    analyzer = WSIAnalyzer(tile_size=256, stride=512)  # Larger stride for faster demo
    results = analyzer.analyze_slide(slide_path)
    
    print(f"\n🎉 Analysis complete! Check ai_results/ directory for outputs")
    print(f"📁 Files created:")
    print(f"   - ai_predictions.csv (raw results)")
    print(f"   - qupath_annotations.csv (for QuPath import)")

if __name__ == "__main__":
    main()
