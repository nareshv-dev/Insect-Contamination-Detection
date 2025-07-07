from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = YOLO("best_model.pt")  # Your trained model

def get_message(label_names):
    messages = {
        "Rice Weevil": """✅ Insect Found: Rice Weevil (Sitophilus oryzae)

🪲 Small beetles that damage stored grains like rice, wheat, and maize.

🥚 Females lay eggs inside grains. Baby insects (larvae) eat the grain from inside.

🔍 Signs of Damage:
• Small holes or powder around grains.
• Grains feel lighter or break easily.

🌡️ Loves warm and humid places – spreads quickly if not controlled.

🛡️ Prevention Tips:
• Sun-dry grains before storing.
• Use airtight containers or sealed bags.
• Add natural repellents like neem or bay leaves.
• Clean storage area regularly.

📦 Store grains in cool, dry places to stay safe!
""",
        "Angoumois grain moth": """✅ Insect Found: Angoumois Grain Moth (Sitotroga cerealella)

🦋 Tiny moths that infest stored grains like rice, wheat, maize, and barley.

🥚 Females lay eggs on grains. Larvae hatch and eat the grain from inside.

🔍 Signs of Damage:
• Grains with tiny exit holes
• Powdery residue or bad smell
• Reduced weight and quality

🌡️ Thrives in warm, humid storage areas – silent but fast spreader.

🛡️ Prevention Tips:
• Dry grains under the sun before storage
• Use sealed, airtight containers
• Keep storage area clean and ventilated
• Use pheromone traps for early detection

📦 Store grains in cool, dry conditions to prevent infestation!""",
        "Indian meal mouth adult": """✅ Insect Found: Indian Meal Moth (Plodia interpunctella)

🦟 Small moths that infest stored food like grains, flour, nuts, cereal, and dried fruits.

🥚 Females lay eggs near food. Larvae spin webs and contaminate food while feeding.

🔍 Signs of Damage:
• Webbing in food containers
• Clumps or foul odor in grains
• Flying moths near kitchen or pantry

🌡️ Active in warm conditions and can spread across stored food quickly.

🛡️ Prevention Tips:
• Store food in airtight glass or plastic containers
• Regularly check and clean pantry shelves
• Discard infested food immediately
• Use bay leaves or traps to repel moths

📦 Keep storage cool, dry, and sealed to stay pest-free!""",
        "Indian meal mouth egg": """✅ Stage: Indian Meal Moth – 🥚 Egg

🦟 Tiny white or grayish eggs, barely visible to the naked eye.

📍 Laid near or directly on stored food like flour, grains, nuts, and dry fruits.

🔢 A single female can lay 100–400 eggs in her lifetime!

🌡️ Eggs hatch in 2–14 days, depending on temperature and humidity.

🔍 Risks:
• Eggs hatch into larvae that spin silk and spoil food
• Often go unnoticed until larval webbing is seen

🛡️ Prevention Tips:
• Inspect new food items before storage
• Store dry goods in sealed, airtight containers
• Clean pantry shelves regularly
• Use natural repellents like cloves or bay leaves

📦 Keeping storage areas cool, dry, and clean prevents egg-laying!""",
        "Khapara bettle": """✅ Insect Found: Khapra Beetle (Trogoderma granarium)

🪳 One of the most destructive pests of stored grains like wheat, rice, maize, and pulses.

🥚 Females lay eggs in cracks or directly on grains. Larvae are hairy and do the most damage.

🔍 Signs of Damage:
• Presence of cast skins and larval hairs
• Grains become powdery, discolored, or hollow
• Infested grains have a musty smell

🌡️ Thrives in hot, dry climates – very hard to eliminate once infested.

🛡️ Prevention Tips:
• Thoroughly clean and disinfect storage areas
• Store grains in airtight containers
• Discard or burn heavily infested stock
• Use high-temperature treatments or fumigation for severe cases

📦 Ensure cool, dry, and sealed storage to stay protected!""",
        "Khapara bettle larva": """✅ Stage: Khapra Beetle – 🐛 Larva

🪳 Most destructive stage of the Khapra Beetle lifecycle.

👁️‍🗨️ Appearance:
• Yellowish-white with brown head
• Covered in dense hairs and bristles
• Grows up to 4–5 mm long

⚠️ Damage Caused:
• Feeds aggressively on grains and pulses
• Leaves behind cast skins and hairs, contaminating food
• Causes grain powdering, discoloration, and foul smell

🧬 Can survive long periods without food, making control difficult

🛡️ Prevention Tips:
• Inspect and clean storage thoroughly before filling
• Use airtight containers or metal bins
• Apply heat or cold treatments to kill larvae
• Avoid mixing old stock with new stock

📦 Maintain dry, clean, and pest-proof grain storage conditions!""",
        "Lesser grain boree": """✅ Insect Found: Lesser Grain Borer (Rhyzopertha dominica)

🪲 A small but highly destructive beetle that bores into stored grains like wheat, rice, maize, and sorghum.

🥚 Females lay eggs on or near grains. Both adults and larvae feed on the grain from inside.

🔍 Signs of Damage:
• Grains are hollowed out and powdery
• Presence of frass (insect waste) and bore holes
• Grain emits a foul, moldy odor

🌡️ Prefers warm, dry conditions and can fly, spreading quickly across storage areas.

🛡️ Prevention Tips:
• Clean storage areas before use
• Store grains in sealed, airtight containers
• Use grain drying to reduce moisture
• Apply fumigation in case of severe infestation

📦 Keep grains cool, dry, and protected to avoid damage!""",
        "Sawtoothed": """✅ Insect Found: Sawtoothed Grain Beetle (Oryzaephilus surinamensis)

🪲 A flat, slender beetle commonly found in stored products like cereals, flour, pasta, dry fruits, and pet food.

🦷 Named for the 6 tiny tooth-like projections on each side of its thorax (like a saw blade!).

🥚 Females lay eggs in food cracks. Larvae and adults feed on the surface of food products.

🔍 Signs of Damage:
• Webbing, clumping, or foul smell in stored goods
• Presence of live beetles crawling in packaging
• Grain dust and broken pieces

🌡️ Loves warm, moist conditions and spreads easily in pantries and storage.

🛡️ Prevention Tips:
• Store items in airtight, pest-proof containers
• Clean shelves and cracks regularly
• Discard infested packages immediately
• Use natural repellents like bay leaves or traps

📦 Keep food storage areas cool, dry, and clean to stay bug-free!""",
        "tobaco bettle": """✅ Insect Found: Tobacco Beetle (Lasioderma serricorne)

🪳 A tiny reddish-brown beetle that infests tobacco products, spices, dried herbs, cereals, books, and even furniture glue.

🥚 Females lay eggs in or near stored products. Larvae are the main culprits — they feed and cause contamination.

🔍 Signs of Damage:
• Holes in tobacco leaves or packaging
• Presence of powdery frass (insect droppings)
• Damaged spices, dried flowers, or herbs

🌡️ Thrives in warm, humid places — very common in storage rooms and warehouses.

🛡️ Prevention Tips:
• Store items in airtight containers
• Use cold treatment (freezing) to kill all stages
• Inspect packages regularly for damage
• Keep storage areas clean and dry

📦 Proper sealing and cool, dry conditions help prevent infestations!"""
    }

    for name in label_names:
        if name in messages:
            return messages[name]
    return "No harmful insects detected!"

@app.route('/', methods=['GET', 'POST'])
def index():
    message = ""
    if request.method == 'POST':
        image = request.files['image']
        image_path = "static/uploaded.jpg"
        image.save(image_path)

        img = cv2.imread(image_path)
        results = model.predict(img, conf=0.25)

        label_names = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                label_names.append(model.names[cls])

        # Save result image
        annotated = results[0].plot()
        result_path = "static/result.jpg"
        cv2.imwrite(result_path, annotated)

        message = get_message(label_names)

        return render_template("index.html", result_image=result_path, message=message)

    return render_template("index.html", result_image=None, message=message)

if __name__== "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
