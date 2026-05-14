/**
 * LocalML Service
 * Performs histogram-based analysis for device-side Computer Vision
 */

export interface LocalAnalysis {
  palette: string[];
  shadeNames: string[];
  complexity: 'Simple' | 'Medium' | 'High';
  temperature: 'Warm' | 'Cool' | 'Neutral';
  /** True if a person/model appears to be wearing the item in the photo */
  hasModel: boolean;
  /**
   * Best-guess category from client-side shape analysis.
   * Used to pre-fill the category dropdown instantly before the backend responds.
   * Values: 'Shoes' | 'Sneakers' | 'Boots' | 'Heels' | 'Sandals' | 'Loafers' |
   *         'Flats' | 'Slides' | 'Top' | 'Dress' | 'Pants' | 'Bag' | '' (unknown)
   */
  guessCategory: string;
  /** Shoe sub-type when guessCategory is 'Shoes' */
  shoeSubtype: string;
}

/**
 * Couture-focused color dictionary for high-precision fashion naming.
 * Expanded to 50+ colors to prevent mismatched nearest-neighbor lookups.
 * RGB values are calibrated to real-world fabric dyes.
 */
const COLOR_MAP: Record<string, [number, number, number]> = {
  // Whites & Neutrals
  "Optical White": [245, 245, 245],
  "Ivory": [255, 255, 240],
  "Bone": [227, 218, 201],
  "Champagne": [247, 231, 206],
  "Cream": [255, 253, 208],
  "Beige": [245, 245, 220],
  "Stone": [215, 208, 199],
  
  // Grays & Blacks
  "Dove Gray": [169, 169, 169],
  "Charcoal": [54, 69, 79],
  "Obsidian": [35, 35, 35],
  "Slate": [112, 128, 144],
  "Jet Black": [10, 10, 10],
  "Silver": [192, 192, 192],
  "Ash": [178, 190, 181],
  "Pewter": [105, 105, 105],

  // Blues & Denim Specifics
  "Raw Indigo": [21, 27, 141],
  "Vintage Indigo": [71, 102, 150],
  "Washed Cobalt": [95, 129, 157],
  "Mid-Wash Blue": [100, 149, 237],
  "Stone Wash": [176, 196, 222],
  "Midnight Navy": [25, 25, 112],
  "Steel Blue": [70, 130, 180],
  "Sky Azure": [135, 206, 235],
  "Deep Teal": [0, 128, 128],
  "Cyan": [0, 255, 255],
  "Turquoise": [64, 224, 208],
  "Baby Blue": [137, 207, 240],
  "Dusty Blue": [136, 157, 175],

  // Greens & Teals
  "Mint": [189, 252, 201], // Calibrated for garment mint (less neon)
  "Sage": [156, 175, 136],
  "Seafoam": [147, 223, 184],
  "Olive": [128, 128, 0],
  "Forest": [34, 139, 34],
  "Emerald": [80, 200, 120],
  "Lime": [191, 255, 0],
  "Kelly Green": [76, 187, 23],
  "Slate Green": [47, 79, 79],
  "Muted Teal": [95, 158, 160],

  // Earth Tones
  "Camel": [193, 154, 107],
  "Cognac": [154, 70, 61],
  "Sand": [194, 178, 128],
  "Taupe": [135, 124, 113], // Calibrated lighter/grayer
  "Coffee": [111, 78, 55],
  "Espresso": [62, 40, 36],
  "Chocolate": [90, 50, 40],
  "Terracotta": [226, 114, 91],
  "Rust": [183, 65, 14],
  "Brown": [139, 69, 19],
  "Mushroom": [186, 171, 160],
  "Khaki": [189, 183, 107],
  "Mocha": [150, 121, 105],

  // Pinks, Purples & Reds
  "Blush": [255, 228, 225],
  "Rosewood": [101, 0, 11],
  "Bordeaux": [109, 7, 26],
  "Crimson": [153, 0, 0],
  "Red": [255, 0, 0],
  "Soft Peony": [255, 192, 203],
  "Hot Pink": [255, 105, 180],
  "Mauve": [224, 176, 255],
  "Dusty Mauve": [180, 140, 150],
  "Lilac": [200, 162, 200],
  "Lavender": [230, 230, 250],
  "Plum": [142, 69, 133],
  "Violet": [138, 43, 226],
  "Coral": [255, 127, 80],
  "Salmon": [250, 128, 114]
};

const getDistance = (c1: [number, number, number], c2: [number, number, number]) => {
  return Math.sqrt(Math.pow(c1[0]-c2[0], 2) + Math.pow(c1[1]-c2[1], 2) + Math.pow(c1[2]-c2[2], 2));
};

const getColorName = (rgbStr: string, excludedNames: Set<string>): string => {
  const match = rgbStr.match(/\d+/g);
  if (!match) return "Neutral";
  const target = match.map(Number) as [number, number, number];
  
  let candidates: { name: string, dist: number }[] = [];

  for (const [name, rgb] of Object.entries(COLOR_MAP)) {
    const dist = getDistance(target, rgb);
    candidates.push({ name, dist });
  }
  
  candidates.sort((a, b) => a.dist - b.dist);
  
  // Try to find a unique name
  for (const cand of candidates) {
    if (!excludedNames.has(cand.name)) {
      // Safety check: If the "unique" option is vastly different from the best option (dist > 30),
      // we prefer accuracy over uniqueness.
      // Example: If best is Brown (dist 5), but Brown is used, and next is Green (dist 80),
      // we should NOT pick Green. We pick Brown again or a variant.
      const bestDist = candidates[0].dist;
      if (cand.dist > bestDist + 30) {
        return candidates[0].name; // Revert to best match
      }
      return cand.name;
    }
  }
  
  return candidates[0].name;
};


// ─── Shoe sub-type detection from silhouette ─────────────────────────────────
/**
 * Uses canvas pixel analysis on a 128×128 greyscale silhouette to classify
 * shoe sub-types without any ML model.
 *
 * Signals used:
 *  • aspectRatio (h/w)   — tall → boots, wide → sneakers/flats
 *  • fillRatio           — low fill → strappy sandals / slides
 *  • soleThickness       — thick bottom band → platform / wedge
 *  • heelColumn          — dense narrow right strip → heels
 *  • toeWidth            — wide front → sneakers/loafers; narrow → heels/oxfords
 */
function classifyShoeSubtype(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number
): string {
  const data = ctx.getImageData(0, 0, W, H).data;

  // Build a binary silhouette: pixel is "shoe" if it differs from the
  // dominant background colour by more than a threshold.
  // Use top-left 8×8 corner as background sample.
  let bgR = 0, bgG = 0, bgB = 0, bgCount = 0;
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const i = (y * W + x) * 4;
      bgR += data[i]; bgG += data[i+1]; bgB += data[i+2];
      bgCount++;
    }
  }
  bgR /= bgCount; bgG /= bgCount; bgB /= bgCount;

  const mask: boolean[][] = Array.from({ length: H }, (_, y) =>
    Array.from({ length: W }, (_, x) => {
      const i = (y * W + x) * 4;
      const dr = data[i] - bgR, dg = data[i+1] - bgG, db = data[i+2] - bgB;
      return Math.sqrt(dr*dr + dg*dg + db*db) > 30;
    })
  );

  // Bounding box of silhouette
  let minX = W, maxX = 0, minY = H, maxY = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (mask[y][x]) {
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
    }
  }
  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  if (bw < 10 || bh < 10) return 'Shoes'; // can't analyse

  const aspectRatio = bh / bw; // > 1.5 = tall (boots), < 0.9 = wide (sneakers)

  // Fill ratio inside bounding box
  let filledPixels = 0;
  for (let y = minY; y <= maxY; y++)
    for (let x = minX; x <= maxX; x++)
      if (mask[y][x]) filledPixels++;
  const fillRatio = filledPixels / (bw * bh);

  // Sole band: bottom 20% of bounding box
  const soleTop = maxY - Math.round(bh * 0.20);
  let soleFilled = 0, soleTotal = 0;
  for (let y = soleTop; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      soleTotal++;
      if (mask[y][x]) soleFilled++;
    }
  }
  const soleFill = soleFilled / Math.max(soleTotal, 1);

  // Heel column: rightmost 18% of bounding box
  const heelLeft = maxX - Math.round(bw * 0.18);
  let heelFilled = 0, heelTotal = 0;
  for (let y = minY; y <= maxY; y++) {
    for (let x = heelLeft; x <= maxX; x++) {
      heelTotal++;
      if (mask[y][x]) heelFilled++;
    }
  }
  const heelDensity = heelFilled / Math.max(heelTotal, 1);

  // Toe width: top 25% of bounding box, measure width of filled pixels
  const toeBottom = minY + Math.round(bh * 0.25);
  let maxToeWidth = 0;
  for (let y = minY; y <= toeBottom; y++) {
    let rowFilled = 0;
    for (let x = minX; x <= maxX; x++) if (mask[y][x]) rowFilled++;
    if (rowFilled > maxToeWidth) maxToeWidth = rowFilled;
  }
  const toeWidthRatio = maxToeWidth / bw;

  // ── Decision rules (ordered from most specific to least) ───────────────
  if (aspectRatio > 2.2)                                      return 'Boots';     // very tall shaft
  if (aspectRatio > 1.4 && fillRatio > 0.55)                 return 'Boots';     // ankle boots
  if (fillRatio < 0.38 && aspectRatio < 1.2)                 return 'Sandals';   // very open / strappy
  if (fillRatio < 0.50 && soleFill > 0.6 && aspectRatio < 1.3) return 'Slides';  // backless open
  if (soleFill > 0.80 && aspectRatio < 0.85)                 return 'Platform Shoes'; // thick flat sole
  if (heelDensity < 0.20 && aspectRatio < 1.0 && fillRatio > 0.55) return 'Flats';   // no heel column
  if (heelDensity > 0.60 && toeWidthRatio < 0.55)            return 'Heels';     // narrow toe + heel
  if (heelDensity > 0.45 && aspectRatio < 1.2)               return 'Heels';
  if (aspectRatio < 1.0 && fillRatio > 0.65 && toeWidthRatio > 0.70) return 'Sneakers'; // chunky/wide
  if (toeWidthRatio < 0.50 && fillRatio > 0.60 && aspectRatio < 1.3) return 'Loafers';  // narrow toe, closed
  if (aspectRatio < 1.15 && fillRatio > 0.60)                return 'Sneakers';  // generic low shoe
  return 'Shoes';
}

// ─── Quick broad-category guess from shape ───────────────────────────────────
/**
 * Returns a broad category hint using aspect ratio, fill, and sole analysis.
 * Shoe detection first (wide, low, distinctive sole), then clothing silhouettes.
 */
function guessBroadCategory(
  img: HTMLImageElement
): { category: string; shoeSubtype: string } {
  const W = 128, H = 128;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  if (!ctx) return { category: '', shoeSubtype: '' };
  ctx.drawImage(img, 0, 0, W, H);

  const data = ctx.getImageData(0, 0, W, H).data;

  // Background colour (corner average)
  let bgR = 0, bgG = 0, bgB = 0;
  const corners = [[0,0],[0,W-4],[H-4,0],[H-4,W-4]];
  for (const [cy, cx] of corners) {
    const i = (cy * W + cx) * 4;
    bgR += data[i]; bgG += data[i+1]; bgB += data[i+2];
  }
  bgR /= 4; bgG /= 4; bgB /= 4;

  // Silhouette fill ratio (all pixels that differ from bg)
  let filled = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const dr = data[i]-bgR, dg = data[i+1]-bgG, db = data[i+2]-bgB;
      if (Math.sqrt(dr*dr+dg*dg+db*db) > 30) filled++;
    }
  }
  const fillRatio = filled / (W * H);

  // Tight bounding box
  let minX = W, maxX = 0, minY = H, maxY = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const dr = data[i]-bgR, dg = data[i+1]-bgG, db = data[i+2]-bgB;
      if (Math.sqrt(dr*dr+dg*dg+db*db) > 30) {
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX <= minX || maxY <= minY) return { category: '', shoeSubtype: '' };

  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  const aspectRatio = bh / bw; // h/w

  // Bottom-strip sole detection (shoes have a dense, wide bottom band)
  const soleTop = maxY - Math.round(bh * 0.18);
  let soleFilled = 0, soleWidth = 0;
  for (let y = soleTop; y <= maxY; y++) {
    let rowStart = -1, rowEnd = -1;
    for (let x = minX; x <= maxX; x++) {
      const i = (y * W + x) * 4;
      const dr = data[i]-bgR, dg = data[i+1]-bgG, db = data[i+2]-bgB;
      if (Math.sqrt(dr*dr+dg*dg+db*db) > 30) {
        if (rowStart === -1) rowStart = x;
        rowEnd = x;
        soleFilled++;
      }
    }
    if (rowEnd > rowStart) soleWidth = Math.max(soleWidth, rowEnd - rowStart);
  }
  const soleSpan = soleWidth / bw; // how wide the sole is relative to bounding box

  // ── Shoe detection heuristic ──────────────────────────────────────────────
  // Shoes: wider than tall (aspect < 1.4), sole spans most of the width, not very tall
  const looksLikeShoe =
    aspectRatio < 1.6 &&
    soleSpan > 0.50 &&
    fillRatio > 0.20 &&
    fillRatio < 0.90;

  if (looksLikeShoe) {
    const subtype = classifyShoeSubtype(ctx, W, H);
    return { category: 'Shoes', shoeSubtype: subtype };
  }

  // ── Clothing shape hints ──────────────────────────────────────────────────
  if (aspectRatio > 2.8) return { category: 'Dress', shoeSubtype: '' };
  if (aspectRatio > 1.8) return { category: 'Dress', shoeSubtype: '' };
  if (aspectRatio > 1.2 && aspectRatio < 1.8) return { category: 'Top', shoeSubtype: '' };
  if (aspectRatio < 0.7) return { category: 'Bag', shoeSubtype: '' };

  return { category: '', shoeSubtype: '' };
}

export const analyzeImageLocally = (base64: string): Promise<LocalAnalysis> => {
  return new Promise((resolve) => {
    const img = new Image();
    img.src = base64;
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) return resolve({ palette: [], shadeNames: [], complexity: 'Simple', temperature: 'Neutral', hasModel: false, guessCategory: '', shoeSubtype: '' });

      const size = 128;
      
      // Draw only center 60% to canvas to remove background
      const srcX = img.width * 0.2;
      const srcY = img.height * 0.2;
      const srcW = img.width * 0.6;
      const srcH = img.height * 0.6;
      
      canvas.width = size;
      canvas.height = size;
      ctx.drawImage(img, srcX, srcY, srcW, srcH, 0, 0, size, size);

      const data = ctx.getImageData(0, 0, size, size).data;
      const bins: Record<string, number> = {};
      let rT = 0, gT = 0, bT = 0;

      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const i = (y * size + x) * 4;
          const r = data[i], g = data[i+1], b = data[i+2];
          
          const distFromCenter = Math.sqrt(Math.pow(x - size/2, 2) + Math.pow(y - size/2, 2));
          const maxDist = size / 2;
          const weight = Math.max(0.1, 1 - (distFromCenter / maxDist));

          rT += r * weight; gT += g * weight; bT += b * weight;
          
          const qr = Math.floor(r / 20) * 20;
          const qg = Math.floor(g / 20) * 20;
          const qb = Math.floor(b / 20) * 20;
          const key = `rgb(${qr},${qg},${qb})`;
          bins[key] = (bins[key] || 0) + weight;
        }
      }

      const sortedBins = Object.entries(bins).sort((a, b) => b[1] - a[1]);
      const palette: string[] = [];
      const rawRgbPalette: [number, number, number][] = [];

      for (const [rgbStr] of sortedBins) {
        if (palette.length >= 5) break;
        
        const match = rgbStr.match(/\d+/g);
        if (!match) continue;
        const currentRgb = match.map(Number) as [number, number, number];
        
        const minDistance = 45; // Increased distinctness threshold
        const isTooSimilar = rawRgbPalette.some(p => getDistance(p, currentRgb) < minDistance);
        
        if (!isTooSimilar) {
          palette.push(rgbStr);
          rawRgbPalette.push(currentRgb);
        }
      }

      const usedNames = new Set<string>();
      const shadeNames = palette.map(color => {
        const name = getColorName(color, usedNames);
        usedNames.add(name);
        return name;
      });

      const avgR = rT / (size * size);
      const avgB = bT / (size * size);
      const temperature = avgR > avgB + 10 ? 'Warm' : (avgB > avgR + 10 ? 'Cool' : 'Neutral');

      // ── Model/person detection via skin-tone pixel counting ──────────────
      // Scan the full image at low resolution for skin-tone pixels.
      const skinCanvas = document.createElement('canvas');
      const skinCtx = skinCanvas.getContext('2d');
      skinCanvas.width = 64;
      skinCanvas.height = 64;
      let hasModel = false;
      if (skinCtx) {
        skinCtx.drawImage(img, 0, 0, 64, 64);
        const skinData = skinCtx.getImageData(0, 0, 64, 64).data;
        let skinPixels = 0;
        const totalPixels = 64 * 64;
        for (let i = 0; i < skinData.length; i += 4) {
          const r = skinData[i], g = skinData[i + 1], b = skinData[i + 2];
          // Broad skin-tone heuristic covering light to deep complexions:
          // R is dominant, not too dark, not too bright (avoids white BG / shadows)
          const isSkin =
            r > 60 && g > 30 && b > 15 &&
            r > g && r > b &&
            (r - g) > 5 &&
            r < 250 && g < 220 && b < 200;
          if (isSkin) skinPixels++;
        }
        // >3% skin pixels → a person is likely present wearing the item
        hasModel = (skinPixels / totalPixels) > 0.03;
      }

      const { category: guessCategory, shoeSubtype } = guessBroadCategory(img);
      resolve({ palette, shadeNames, complexity: 'Medium', temperature, hasModel, guessCategory, shoeSubtype });
    };
    img.onerror = () => {
      resolve({ palette: [], shadeNames: [], complexity: 'Simple', temperature: 'Neutral', hasModel: false, guessCategory: '', shoeSubtype: '' });
    };
  });
};
