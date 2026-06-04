from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import torch

app = Flask(__name__)

# Optimize CPU thread count for YOLO inference
torch.set_num_threads(2)

model = YOLO("best_model.pt")  # Your trained model

# Warm up the model at startup to eliminate cold-start latency on the first request
print("Warming up YOLO model...")
dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
model.predict(dummy_img, conf=0.25, verbose=False)
print("YOLO model warmed up successfully!")

PEST_DATABASE = {
    "Rice Weevil": {
        "display_name": "Rice Weevil",
        "scientific_name": "Sitophilus oryzae",
        "severity_score": 2,
        "severity_label": "Medium",
        "color_class": "medium-risk",
        "description": "Small beetles that cause severe damage to stored grains like rice, wheat, and maize.",
        "details": "Females lay eggs inside the grain kernel, and the emerging larvae feed on the grain from the inside out, making them highly destructive.",
        "damage_signs": [
            "Small entry/exit holes in grains",
            "Powdery residue (frass) around storage containers",
            "Grains feel unusually light or break easily"
        ],
        "prevention_tips": [
            "Sun-dry grains thoroughly before storage",
            "Use airtight containers or sealed bags",
            "Add natural repellents like neem or bay leaves",
            "Regularly clean and sanitize the storage area"
        ]
    },
    "Angoumois grain moth": {
        "display_name": "Angoumois Grain Moth",
        "scientific_name": "Sitotroga cerealella",
        "severity_score": 2,
        "severity_label": "Medium",
        "color_class": "medium-risk",
        "description": "Tiny moths that infest stored grains, particularly rice, wheat, maize, and barley.",
        "details": "Females lay eggs directly on the grains. The hatched larvae bore inside and feed on the endosperm, leaving empty husks.",
        "damage_signs": [
            "Tiny round exit holes on grain surfaces",
            "Foul, musty odor or powdery residue",
            "Reduced grain weight and quality"
        ],
        "prevention_tips": [
            "Dry grains under the sun before storage",
            "Store in clean, airtight containers",
            "Keep storage area well-ventilated and cool",
            "Deploy pheromone traps for early detection"
        ]
    },
    "Indian meal mouth adult": {
        "display_name": "Indian Meal Moth (Adult)",
        "scientific_name": "Plodia interpunctella",
        "severity_score": 2,
        "severity_label": "Medium",
        "color_class": "medium-risk",
        "description": "Small moths that infest stored pantry products such as grains, flour, nuts, cereals, and dried fruits.",
        "details": "While adults do not feed, they lay hundreds of eggs near food sources, leading to rapid larval infestations.",
        "damage_signs": [
            "Adult moths flying in a zig-zag pattern near kitchens",
            "Webbing on the surface of food packages",
            "Clumping of food products"
        ],
        "prevention_tips": [
            "Store all dry foods in airtight glass or plastic containers",
            "Regularly inspect and vacuum pantry shelves",
            "Immediately discard infested food",
            "Use pantry moth traps or natural repellents like bay leaves"
        ]
    },
    "Indian meal mouth egg": {
        "display_name": "Indian Meal Moth (Egg)",
        "scientific_name": "Plodia interpunctella",
        "severity_score": 1,
        "severity_label": "Low",
        "color_class": "low-risk",
        "description": "Microscopic white or grayish eggs laid near or directly on stored food.",
        "details": "A single female can lay up to 400 eggs. They hatch in 2–14 days depending on temperature, initiating the destructive larval phase.",
        "damage_signs": [
            "Almost invisible to the naked eye",
            "Often detected only after larval webbing starts to appear",
            "Risk of rapid hatching in warm environments"
        ],
        "prevention_tips": [
            "Carefully inspect newly purchased dry goods before storing",
            "Keep dry goods sealed in tight glass or metal containers",
            "Store in cool, dry areas to inhibit hatching",
            "Keep shelves clean and vacuumed"
        ]
    },
    "Indian meal mouth larva": {
        "display_name": "Indian Meal Moth (Larva)",
        "scientific_name": "Plodia interpunctella",
        "severity_score": 2,
        "severity_label": "Medium",
        "color_class": "medium-risk",
        "description": "The highly destructive caterpillar stage of the Indian Meal Moth.",
        "details": "Cream-colored larvae spin dense silk webbing as they feed, contaminating grains, flour, and dry fruits with droppings and webs.",
        "damage_signs": [
            "Silken webbing clumping grains together",
            "Crawling caterpillars in cupboards or containers",
            "Fecal pellets and cast skins in food"
        ],
        "prevention_tips": [
            "Immediately isolate and discard infested items",
            "Store food in robust, sealed containers",
            "Deep-clean cracks and crevices in shelves",
            "Use pheromone traps to capture adult moths"
        ]
    },
    "Khapara bettle": {
        "display_name": "Khapra Beetle",
        "scientific_name": "Trogoderma granarium",
        "severity_score": 3,
        "severity_label": "High",
        "color_class": "high-risk",
        "description": "One of the world's most destructive and dreaded pests of stored grains, seeds, and husks.",
        "details": "Known for its ability to survive long periods without food and resistance to many insecticides. Larvae feed aggressively, leaving powdery ruins.",
        "damage_signs": [
            "Grains hollowed out into powder",
            "Abundant hairy cast skins and larval hairs",
            "Musty, unpleasant smell in storage"
        ],
        "prevention_tips": [
            "Clean and disinfect storage bins thoroughly before refilling",
            "Ensure strict quarantine and sanitation",
            "Apply high-temperature treatment if infested",
            "Store in absolute airtight conditions"
        ]
    },
    "Khapara bettle larva": {
        "display_name": "Khapra Beetle (Larva)",
        "scientific_name": "Trogoderma granarium",
        "severity_score": 3,
        "severity_label": "High",
        "color_class": "high-risk",
        "description": "The most destructive lifecycle stage of the notorious Khapra Beetle.",
        "details": "Larvae are yellowish-white and covered in dense, irritating hairs. They feed relentlessly and shed skins, causing severe grain contamination.",
        "damage_signs": [
            "Presence of dense, bristly hairs and cast skins",
            "Aggressive powdering and discoloration of grains",
            "Foul, musty odor"
        ],
        "prevention_tips": [
            "Avoid mixing old stock with new stock",
            "Use metal bins or thick airtight containers",
            "Apply heat treatment (above 60°C) or freezing to destroy larvae",
            "Strictly sanitize all storage equipment"
        ]
    },
    "Lesser grain boree": {
        "display_name": "Lesser Grain Borer",
        "scientific_name": "Rhyzopertha dominica",
        "severity_score": 3,
        "severity_label": "High",
        "color_class": "high-risk",
        "description": "A small, highly destructive beetle that bores deep inside stored grain kernels.",
        "details": "Both adults and larvae feed aggressively. Their strong jaws allow them to chew through wood, paper, and plastic packaging.",
        "damage_signs": [
            "Hollowed-out grains and fine floury dust (frass)",
            "Distinct bore holes in grain seeds",
            "Strong, sweetish or musty odor"
        ],
        "prevention_tips": [
            "Clean storage facilities thoroughly before storing new grains",
            "Ensure grain moisture is below 12% to restrict growth",
            "Store in heavy-duty airtight containers",
            "Use fumigation or extreme temperature controls if needed"
        ]
    },
    "Sawtoothed": {
        "display_name": "Sawtoothed Grain Beetle",
        "scientific_name": "Oryzaephilus surinamensis",
        "severity_score": 1,
        "severity_label": "Low",
        "color_class": "low-risk",
        "description": "A flat, slender beetle commonly found in cereals, flour, dried fruits, and pet food.",
        "details": "Named for the six saw-like teeth on its thorax. They feed on broken kernels and grain dust rather than whole grains, but multiply rapidly.",
        "damage_signs": [
            "Live flat brown beetles crawling in packages",
            "Grain dust and small broken pieces",
            "Slight warming of infested grain due to insect activity"
        ],
        "prevention_tips": [
            "Store food in airtight plastic, glass, or metal containers",
            "Regularly clean pantry shelves and vacuum cracks",
            "Discard infested products immediately",
            "Store dry goods in cool, dry conditions"
        ]
    },
    "tobaco bettle": {
        "display_name": "Tobacco Beetle",
        "scientific_name": "Lasioderma serricorne",
        "severity_score": 2,
        "severity_label": "Medium",
        "color_class": "medium-risk",
        "description": "A tiny, oval reddish-brown beetle that infests tobacco, spices, herbs, and books.",
        "details": "Also known as the Cigarette Beetle. Larvae do the most damage by burrowing through and contaminating stored materials.",
        "damage_signs": [
            "Pinhead-sized holes in tobacco leaves, spices, or cardboard",
            "Powdery droppings (frass) at the bottom of containers",
            "Damaged dried herbs or books"
        ],
        "prevention_tips": [
            "Store spices and dried goods in airtight glass jars",
            "Freeze suspect packages for 6 days at -18°C to kill eggs/larvae",
            "Keep storage rooms dry and clean",
            "Regularly inspect stored books and dried collections"
        ]
    }
}

def get_message(label_names):
    # Backward compatibility for plain-text message
    for name in label_names:
        if name in PEST_DATABASE:
            pest = PEST_DATABASE[name]
            return f"✅ Insect Found: {pest['display_name']} ({pest['scientific_name']})\n\n{pest['description']}\n\n{pest['details']}"
    return "No harmful insects detected!"

def get_risk_assessment(label_names):
    # Backward compatibility for plain-text risk assessment
    risk_summary = get_risk_assessment_structured(label_names)
    return f"⚠️ Risk Assessment Summary:\nRisk Level: {risk_summary['emoji']} {risk_summary['category'].upper()}\nAverage Severity Score: {risk_summary['score']:.2f}/3.0"

def get_detected_pests(label_names):
    detected = []
    for name in label_names:
        if name in PEST_DATABASE:
            detected.append(PEST_DATABASE[name])
    return detected

def get_risk_assessment_structured(label_names):
    total_risk = 0
    count = 0

    for name in label_names:
        if name in PEST_DATABASE:
            total_risk += PEST_DATABASE[name]["severity_score"]
            count += 1

    if count == 0:
        risk_score = 0
        category = "No Risk"
        emoji = "🟢"
        color_class = "no-risk"
        bg_color = "#00b894"
        description = "Your grains are clean! No harmful pests were detected in the analyzed sample."
    else:
        risk_score = total_risk / count
        if risk_score <= 1.5:
            category = "Low"
            emoji = "🟢"
            color_class = "low-risk"
            bg_color = "#00b894"
            description = "Minor pest activity detected. Monitor storage conditions and ensure proper ventilation."
        elif risk_score <= 2.5:
            category = "Medium"
            emoji = "🟡"
            color_class = "medium-risk"
            bg_color = "#feca57"
            description = "Moderate pest infestation detected. Action should be taken soon to prevent further spread."
        else:
            category = "High"
            emoji = "🔴"
            color_class = "high-risk"
            bg_color = "#ff6b6b"
            description = "Severe pest infestation detected! Immediate action, cleaning, or fumigation is highly recommended to protect your grains."

    return {
        "score": round(risk_score, 2),
        "score_percentage": int((risk_score / 3.0) * 100),
        "category": category,
        "emoji": emoji,
        "color_class": color_class,
        "bg_color": bg_color,
        "description": description
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    message = ""
    risk_assessment = ""
    detected_pests = []
    overall_risk = get_risk_assessment_structured([]) # Default empty risk state

    if request.method == 'POST':
        image = request.files['image']
        
        # Read image directly from memory to bypass disk write/read cycles
        file_bytes = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        results = model.predict(img, conf=0.25)

        label_names = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                label_names.append(model.names[cls])

        label_names = list(set(label_names))  # Remove duplicates

        # Save result image
        annotated = results[0].plot()
        result_path = "static/result.jpg"
        cv2.imwrite(result_path, annotated)

        # Get both risk assessment and detailed message
        message = get_message(label_names)
        risk_assessment = get_risk_assessment(label_names)
        
        detected_pests = get_detected_pests(label_names)
        overall_risk = get_risk_assessment_structured(label_names)

        return render_template("index.html", 
                             result_image=result_path, 
                             risk_assessment=risk_assessment,
                             message=message,
                             detected_pests=detected_pests,
                             overall_risk=overall_risk,
                             pest_database=PEST_DATABASE)

    return render_template("index.html", 
                          result_image=None, 
                          risk_assessment=risk_assessment,
                          message=message,
                          detected_pests=detected_pests,
                          overall_risk=overall_risk,
                          pest_database=PEST_DATABASE)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
