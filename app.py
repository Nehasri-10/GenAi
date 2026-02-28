# ==========================================================
# CLEAN INSTALL (Stable Versions)
# ==========================================================   (cell1)

!pip uninstall -y transformers diffusers accelerate peft huggingface_hub safetensors controlnet_aux timm xformers > /dev/null 2>&1
!pip uninstall -y datasets gradio sentence-transformers
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install groq
!pip install -q \
diffusers==0.25.0 \
transformers==4.36.2 \
accelerate==0.30.0 \
peft==0.10.0 \
huggingface_hub==0.20.3 \
safetensors==0.4.2 \
controlnet_aux==0.0.6 \
timm \
xformers \
opencv-python \
Pillow \
scikit-image \
pandas \
numpy \
requests \
flask \
flask-cors \
pyngrok




#----------------cell2
from groq import Groq

GROQ_API_KEY = "API_KEY"

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.1-8b-instant"

models = groq_client.models.list()
for m in models.data:
    print(m.id)




## IMPORTS
# ==========================================================   (cell3)

import torch
from contextlib import nullcontext

from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel
)

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline as hf_pipeline
)

# ==========================================================
# DEVICE SETUP
# ==========================================================

COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if COMPUTE_DEVICE == "cuda" else torch.float32
DETECTOR_DEVICE = 0 if torch.cuda.is_available() else -1
autocast = torch.autocast("cuda") if COMPUTE_DEVICE == "cuda" else nullcontext()

print("Device:", COMPUTE_DEVICE)

# ==========================================================
# DETR (Object Detection)
# ==========================================================

print("Loading DETR...")
detector = hf_pipeline(
    "object-detection",
    model="facebook/detr-resnet-101",
    device=DETECTOR_DEVICE
)
print("DETR ready ✅")

# ==========================================================
# CONTROLNET
# ==========================================================

print("Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=DTYPE
).to(COMPUTE_DEVICE)

print("ControlNet ready ✅")

# ==========================================================
# STABLE DIFFUSION
# ==========================================================

print("Loading Stable Diffusion...")
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    controlnet=controlnet,
    torch_dtype=DTYPE
).to(COMPUTE_DEVICE)

pipe.enable_attention_slicing()

print("Diffusion ready ✅")

# ==========================================================
# FLAN-T5 (Text reasoning + compression)
# ==========================================================

print("Loading Flan-T5...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

llm_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-large"
).to("cpu")  # Keep on CPU to save GPU memory

print("Flan ready ✅")

# ==========================================================
# TINYLLAMA CHATBOT
# ==========================================================

print("Loading TinyLlama Chatbot...")

chat_tokenizer = AutoTokenizer.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

chat_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=DTYPE,
    device_map="auto"
)

print("Chatbot ready ✅")




# ==========================================================
  # COMMON IMPORTS (Run After Runtime Restart)
  # ==========================================================   (cell4)

  import base64
  import io
  import re
  import threading
  import urllib.parse
  import socket

  import torch
  import cv2
  import numpy as np
  import pandas as pd

  from PIL import Image
  from contextlib import nullcontext
  from skimage.feature import canny as sk_canny

  from flask import Flask, request, jsonify
  from flask_cors import CORS
  from pyngrok import ngrok, conf
  # ═════════════════════════════════════════════════════════════════════
  # UTILITY FUNCTIONS
  # ═════════════════════════════════════════════════════════════════════
  amazon_df = pd.read_csv('/content/amazon.csv')
  ikea_df = pd.read_csv('/content/ikea.csv')
  furniture_df = pd.read_excel('/content/kaggle_furniture.xlsx')

  def b64_to_pil(b64: str) -> Image.Image:
      if "," in b64:
          b64 = b64.split(",")[1]
      return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

  def pil_to_b64(img: Image.Image) -> str:
      buf = io.BytesIO()
      img.save(buf, format="PNG")
      return base64.b64encode(buf.getvalue()).decode("utf-8")

  def preprocess_image(path: str) -> Image.Image:
      return Image.open(path).convert("RGB").resize((512, 512))
  def generate_canny(image: Image.Image) -> Image.Image:
      arr = np.array(image)
      gray = np.mean(arr, axis=2) if arr.ndim == 3 else arr
      edges = sk_canny(gray, low_threshold=50, high_threshold=150)
      return Image.fromarray((edges * 255).astype(np.uint8))
  def chat_reasoning(user_text, geometry, room_type):

      prompt = f"""
      You are an expert interior designer.
      Room type: {room_type}
      Room geometry: {geometry}
      User modification request:
      {user_text}

      STRICT RULES:
      - Preserve layout
      - Preserve furniture placement
      - Do not include placeholders
      - Do not repeat the request
      - Only describe visual changes

      Return EXACTLY:

      REFINE_PROMPT:
    <one concise visual description under 40 words>

    SPACE_SUGGESTIONS:
    - suggestion 1
    - suggestion 2
    - suggestion 3
    - suggestion 4
    """

      response = groq_client.chat.completions.create(
          model=GROQ_MODEL,
          messages=[
              {"role": "system", "content": "You are a professional interior design assistant."},
              {"role": "user", "content": prompt}
          ],
          temperature=0.2,
          max_tokens=300
      )

      return response.choices[0].message.content
  def generate_llm_text(prompt: str, max_length: int = 400) -> str:
      inputs = tokenizer(prompt, return_tensors="pt")
      with torch.no_grad():
          outputs = llm_model.generate(**inputs, max_length=max_length)
      return tokenizer.decode(outputs[0], skip_special_tokens=True)

  def compress_for_diffusion(text: str) -> str:
      prompt = f"""
      Summarize this interior design plan into a short
      diffusion-friendly description under 60 words.
      Focus only on visual changes.
      {text}
      """
      return generate_llm_text(prompt, max_length=120)

  def generate_image(base_image: Image.Image, concept_text: str) -> Image.Image:
      canny_image = generate_canny(base_image)
      positive_prompt = f"""
      Ultra realistic DSLR interior photograph,
      preserve structure and layout,
      {concept_text}
      """
      negative_prompt = "cartoon, anime, distorted layout, unrealistic lighting"
      with autocast:
          result = pipe(
              prompt=positive_prompt,
              negative_prompt=negative_prompt,
              image=base_image,
              control_image=canny_image,
              strength=0.75,
              guidance_scale=8.5,
              num_inference_steps=25,
              controlnet_conditioning_scale=0.6
          )
      return result.images[0]

  def refine_image_fn(current_pil, current_concept, refinement_text, room_type):

      geometry = coordinator.space.analyze(current_pil)

      chatbot_response = chat_reasoning(
          refinement_text,
          geometry,
          room_type
      )

      try:
          refine_part = chatbot_response.split("REFINE_PROMPT:")[1].split("SPACE_SUGGESTIONS:")[0].strip()
          suggestions_part = chatbot_response.split("SPACE_SUGGESTIONS:")[1].strip()
      except:
          refine_part = refinement_text
          suggestions_part = "No structured suggestions generated."

      canny_image = generate_canny(current_pil)

      positive_prompt = f"""
      Ultra realistic DSLR interior photograph,
      preserve structure and layout,
      apply only these changes:
      {refine_part}
      """

      negative_prompt = "cartoon, anime, distorted layout, remove furniture, unrealistic lighting"

      with autocast:
          result = pipe(
              prompt=positive_prompt,
              negative_prompt=negative_prompt,
              image=current_pil,
              control_image=canny_image,
              strength=0.6,
              guidance_scale=8.5,
              num_inference_steps=25,
              controlnet_conditioning_scale=0.7
          )

      new_image = result.images[0]

      new_concept = current_concept + "\n" + refine_part

      return new_image, new_concept, suggestions_part

  # ══════════════════════════════════════════════════════
  # AGENTS
  # ═════════════════════════════════════════════════════════════════════

  class DesignStrategistAgent:
      STYLE_MAP = {
          "Modern":             {"wall_color": "warm beige walls",            "materials": "matte black accents, light wood furniture", "patterns": "minimal patterns, clean lines",         "lighting": "warm recessed ceiling lights",        "decor": "abstract art frames"},
          "Indian Traditional": {"wall_color": "deep maroon or mustard walls", "materials": "teak wood furniture, brass decor",           "patterns": "floral or ethnic patterned textiles",   "lighting": "warm yellow lamps",                   "decor": "traditional framed artwork"},
          "Scandinavian":       {"wall_color": "pure white walls",             "materials": "light oak wood, neutral fabrics",            "patterns": "minimal geometric rug",                 "lighting": "soft diffused daylight",              "decor": "indoor green plants"},
          "Luxury":             {"wall_color": "taupe or cream walls",         "materials": "velvet upholstery, marble accents",          "patterns": "rich layered textiles",                 "lighting": "layered ambient + pendant lighting",  "decor": "large statement artwork"},
          "Minimal":            {"wall_color": "off white walls",              "materials": "simple wood furniture",                      "patterns": "very minimal patterns",                 "lighting": "soft natural lighting",               "decor": "limited decor pieces"},
          "Bohemian":           {"wall_color": "warm earthy tones",            "materials": "rattan, wood, mixed textures",               "patterns": "colorful layered textiles",             "lighting": "warm hanging lights",                 "decor": "plants and artistic decor"},
      }
      VIBE_EFFECT = {
          "Calm":    "soft neutral tones, diffused lighting",
          "Bold":    "high contrast colors, dramatic lighting",
          "Cozy":    "warm ambient lighting, soft textiles, layered fabrics",
          "Elegant": "refined finishes, subtle luxury details",
      }

      def generate_concept(self, style: str, purpose: str, vibe: str) -> str:
          s = self.STYLE_MAP.get(style, self.STYLE_MAP["Modern"])
          vibe_prompt = self.VIBE_EFFECT.get(vibe, "")
          return (
              f"Room Purpose: {purpose}\n"
              f"Desired Vibe: {vibe}\n\n"
              f"{vibe_prompt},\n"
              f"Change wall color to {s['wall_color']},\n"
              f"use {s['materials']},\n"
              f"add {s['patterns']},\n"
              f"install {s['lighting']},\n"
              f"include {s['decor']}"
          )


  class SpaceOptimizationAgent:
      def analyze(self, image: Image.Image) -> dict:
          w, h = image.size
          area_score = (w * h) / 50000
          if area_score < 3:
              room_size, walking_space = "Compact", 0.8
          elif area_score < 6:
              room_size, walking_space = "Medium", 1.1
          else:
              room_size, walking_space = "Spacious", 1.5
          return {"room_size": room_size, "estimated_walking_space_m": walking_space, "is_space_sufficient": walking_space >= 0.9}

      def suggest_layout(self, room_type: str) -> list:
          layout_map = {
              "Bedroom":     ["Place bed against longest wall", "Maintain 0.9m minimum walking clearance", "Keep wardrobe near corner to maximize space", "Avoid blocking window light"],
              "Living Room": ["Place sofa facing focal wall or TV unit", "Maintain central walking path", "Keep coffee table within 45cm of sofa", "Avoid blocking balcony or window light"],
              "Study Room":  ["Place study desk near window for natural light", "Maintain ergonomic chair spacing", "Keep bookshelf against wall", "Maintain clutter-free walking area"],
              "Office":      ["Position desk facing entry or window", "Maintain 1m chair movement clearance", "Keep storage cabinets against walls", "Avoid blocking electrical outlets"],
              "Kitchen":     ["Maintain work triangle between sink, stove, fridge", "Keep minimum 1m circulation space", "Avoid blocking ventilation areas", "Place dining near natural light"],
          }
          return layout_map.get(room_type, ["Optimize layout for comfort and circulation"])


  # ── Shopping constants ────────────────────────────────────────────────

  JUNK_TOKENS = {
      "bedroom","living room","study room","office","kitchen","furniture","answer",
      "room","type","item","accents","materials","patterns","lighting","decor","style",
      "design","concept","color","scheme","wall","floor","ceiling","modern","minimal",
      "luxury","traditional","scandinavian","bohemian","indian","warm","cool","ambient",
      "natural","artificial","soft","bright","dark","light","matte","glossy","texture",
      "finish","wood","metal","glass","fabric","velvet","marble","oak","teak","brass",
      "black","white","beige","yes","no","none","null","n/a","na","list",
      "the","and","for","with","from","into","onto",
  }

  SYNONYM_MAP = {
      "couch":"sofa","loveseat":"sofa","sectional":"sofa","settee":"sofa",
      "armchair":"chair","accent chair":"chair","stool":"chair",
      "bookcase":"bookshelf","book shelf":"bookshelf",
      "cupboard":"wardrobe","closet":"wardrobe","almirah":"wardrobe",
      "dresser drawer":"dresser","chest of drawers":"dresser",
      "rug":"carpet","area rug":"carpet",
      "drapes":"curtains","drape":"curtains","curtain":"curtains",
      "floor lamp":"table lamp","desk lamp":"table lamp","lamp":"table lamp",
      "tv stand":"tv unit","tv cabinet":"tv unit","media unit":"tv unit","entertainment unit":"tv unit",
      "side table":"nightstand","bedside table":"nightstand",
      "study table":"desk","work desk":"desk","writing table":"desk",
      "dining set":"dining table",
      "plant":"indoor plant","potted plant":"indoor plant","indoor plants":"indoor plant",
      "picture frame":"art frame","wall art":"art frame","painting":"art frame",
  }

  # ═════════════════════════════════════════════════════════════════════
  # SHOPPING + BUDGET ENGINE (FINAL STABLE VERSION)
  # ═════════════════════════════════════════════════════════════════════

  import re
  import numpy as np

  DATASET_PRICE_MIN = 500
  DATASET_PRICE_MAX = 500000

  # Room-based strict filtering
  BASE_ROOM_ITEMS = {
      "Bedroom":     {"bed","wardrobe","nightstand","dresser","table lamp","carpet","curtains","art frame"},
      "Living Room": {"sofa","tv unit","coffee table","carpet","curtains","chair","art frame","indoor plant"},
      "Study Room":  {"desk","chair","bookshelf","office chair","table lamp","carpet","curtains","indoor plant"},
      "Office":      {"desk","office chair","cabinet","bookshelf","table lamp"},
      "Kitchen":     {"dining table","bar stool","refrigerator","cabinet"},
  }

  STYLE_MULTIPLIER = {
      "Modern":1.0,
      "Luxury":1.4,
      "Minimal":0.9,
      "Indian Traditional":1.2,
      "Bohemian":1.1,
      "Scandinavian":1.0,
  }

  FALLBACK_PRICE_RANGE = {
      "bed":(18000,30000),
      "wardrobe":(15000,60000),
      "sofa":(20000,50000),
      "curtains":(2000,8000),
      "carpet":(3000,20000),
      "chair":(1500,8000),
      "desk":(5000,25000),
      "coffee table":(4000,20000),
      "tv unit":(8000,35000),
      "cabinet":(10000,40000),
      "bookshelf":(6000,20000),
      "office chair":(4000,15000),
      "dining table":(15000,50000),
      "bar stool":(2000,8000),
      "refrigerator":(15000,60000),
      "nightstand":(3000,12000),
      "dresser":(8000,25000),
      "table lamp":(1500,5000),
      "indoor plant":(500,3000),
      "art frame":(800,5000),
  }


  def normalize_item(text):
      text = text.lower().strip()
      text = re.sub(r"[^a-zA-Z\s]", "", text)
      text = re.sub(r"\s+", " ", text)
      return text


  def match_price_from_dataset(item):
      prices = []

      try:
          matches = amazon_df[amazon_df["title"].str.contains(item, case=False, na=False)]
          prices += pd.to_numeric(matches["price"], errors="coerce").dropna().tolist()
      except:
          pass

      try:
          matches = ikea_df[ikea_df["name"].str.contains(item, case=False, na=False)]
          prices += pd.to_numeric(matches["price"], errors="coerce").dropna().tolist()
      except:
          pass

      prices = [p for p in prices if DATASET_PRICE_MIN < p < DATASET_PRICE_MAX]

      if not prices:
          return None

      return float(np.median(prices[:20]))


  def get_item_price(item, style):
      multiplier = STYLE_MULTIPLIER.get(style, 1.0)

      dataset_price = match_price_from_dataset(item)
      if dataset_price:
          return round(dataset_price * multiplier)

      if item in FALLBACK_PRICE_RANGE:
          lo, hi = FALLBACK_PRICE_RANGE[item]
          return round(((lo + hi) / 2) * multiplier)

      return None

  def scale_to_budget(price_map, budget_min, budget_max):

      total = sum(price_map.values())

      if total <= 0:
          return price_map

      # If within range, just return
      if budget_min <= total <= budget_max:
          return price_map

      target = (budget_min + budget_max) / 2
      scale_factor = target / total

      scaled = {}
      for item, price in price_map.items():
          scaled[item] = max(1, int(round(price * scale_factor)))

      return scaled

  class ShoppingAssistantAgent:

      LABEL_NORMALIZATION = {
          "couch":"sofa",
          "chair":"chair",
          "dining table":"dining table",
          "bed":"bed",
          "tv":"tv unit",
          "potted plant":"indoor plant",
          "refrigerator":"refrigerator"
      }

      FURNITURE_CLASSES = [
          "couch","chair","dining table",
          "bed","tv","potted plant","refrigerator"
      ]

      def detect_furniture(self, image):

          detections = detector(image)
          items = []

          for obj in detections:
              if obj["score"] >= 0.5 and obj["label"] in self.FURNITURE_CLASSES:
                  items.append(self.LABEL_NORMALIZATION[obj["label"]])

          return list(set(items))


      def extract_furniture_from_concept(self, concept_text, room_type):
        prompt = f"""
        Extract ONLY purchasable furniture items appropriate for a {room_type}.
        Return comma separated nouns only.
        Ignore colors, styles, materials.
        Text:
        {concept_text}
        """

        response = groq_client.chat.completions.create(
          model=GROQ_MODEL,
          messages=[
              {"role": "system", "content": "Extract furniture items only."},
              {"role": "user", "content": prompt}
          ],
          temperature=0.2,
          max_tokens=100
          )
        text = response.choices[0].message.content.lower()
        return [normalize_item(x) for x in text.replace("\n", ",").split(",") if normalize_item(x)]


      def generate_store_links(self, item, style):

          encoded = urllib.parse.quote(f"{style} {item}")

          return {
              "Amazon": f"https://www.amazon.in/s?k={encoded}",
              "IKEA": f"https://www.ikea.com/in/en/search/?q={encoded}",
              "Pepperfry": f"https://www.pepperfry.com/site_product/search?q={encoded}"
          }


      def generate_shopping_and_budget(self, image, style, concept_text, room_type, budget_min, budget_max):

          visual_items = list(self.detect_furniture(image))
          semantic_items = list(self.extract_furniture_from_concept(concept_text, room_type))
          baseline = list(BASE_ROOM_ITEMS.get(room_type, []))
          # Merge everything
          combined = list(set(visual_items + semantic_items + baseline))
          # Strict room filtering
          allowed = BASE_ROOM_ITEMS.get(room_type, set())
          filtered = [i for i in combined if i in allowed]

          if not filtered:
              return [], 0

          price_map = {}

          for item in filtered:
              price = get_item_price(item, style)
              if price and price > 0:
                  price_map[item] = price

          if not price_map:
              return [], 0

          scaled_prices = scale_to_budget(price_map, budget_min, budget_max)

          results = []

          for item, price in scaled_prices.items():
              results.append({
                  "item": item,
                  "price": int(round(price)),
                  "stores": self.generate_store_links(item, style)
              })

          return results, int(round(sum(scaled_prices.values())))

  class InteriorCoordinator:
      def __init__(self):
          self.design = DesignStrategistAgent()
          self.space  = SpaceOptimizationAgent()
          self.shop   = ShoppingAssistantAgent()

      def run(self, image, style, room_type, purpose, vibe):
          concept  = self.design.generate_concept(style, purpose, vibe)
          geometry = self.space.analyze(image)
          layout   = self.space.suggest_layout(room_type)
          return concept, geometry, layout


  coordinator = InteriorCoordinator()
  print("All agents initialised ✅")




# ═════════════════════════════════════════════════════════════════════
# FLASK APP
# ═════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data       = request.get_json(force=True)
        b64_image  = data.get("image", "")
        style      = data.get("style", "Modern")
        room_type  = data.get("room_type", "Living Room")
        purpose    = data.get("purpose", "Relaxation")
        vibe       = data.get("vibe", "Cozy")
        budget_min = float(data.get("budget_min", 100000))
        budget_max = float(data.get("budget_max", 300000))

        if not b64_image:
            return jsonify({"error": "No image provided"}), 400

        base_image = b64_to_pil(b64_image).resize((512, 512))

        # Run coordinator
        concept, geometry, layout = coordinator.run(base_image, style, room_type, purpose, vibe)

        # Generate redesigned image
        generated_image = generate_image(base_image, concept)

        # Shopping + budget
        shopping_results, total_budget = coordinator.shop.generate_shopping_and_budget(
            generated_image, style, concept, room_type, budget_min, budget_max
        )

        return jsonify({
            "image":        pil_to_b64(generated_image),
            "depth":        None,
            "concept":      concept.strip(),
            "layout":       layout,
            "shopping":     shopping_results,
            "total_budget": total_budget,
            "geometry":     geometry,
        }), 200

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/refine", methods=["POST"])
def refine():
    try:
        data            = request.get_json(force=True)
        b64_image       = data.get("image", "")
        refinement_text = data.get("refinement_text", "")
        current_concept = data.get("concept", "")
        room_type       = data.get("room_type", "Living Room")

        if not b64_image or not refinement_text:
            return jsonify({"error": "image and refinement_text required"}), 400

        current_pil = b64_to_pil(b64_image)

        new_image, new_concept, suggestions = refine_image_fn(
            current_pil,
            current_concept,
            refinement_text,
            room_type
        )

        return jsonify({
            "image": pil_to_b64(new_image),
            "concept": new_concept.strip(),
            "space_suggestions": suggestions
        }), 200

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500




def start_flask():
    app.run(host="0.0.0.0", port=5000, use_reloader=False, debug=False)

# Configure ngrok
conf.get_default().auth_token = "API_KEY"
ngrok.kill()   # kill any stale tunnels

# Start Flask in background thread
threading.Thread(target=start_flask, daemon=True).start()

# Open public tunnel
public_url = ngrok.connect(5000, bind_tls=True).public_url

print("\n" + "═" * 60)
print("  ✅  InteriorAI Pro backend is LIVE!")
print(f"  🌐  Public URL : {public_url}")
print()
print("  📋  In your frontend HTML, set:")
print(f'      const BACKEND_URL = "{public_url}";')
print("      OR paste it into the Backend URL field in the header")
print("═" * 60 + "\n")
