"""Plant-disease knowledge base: symptoms, causes, treatment and prevention.

Covers the 38 PlantVillage classes (26 diseases + 12 healthy). The guidance is general,
educational information compiled from common extension-service advice. It is NOT a
substitute for a local agronomist: product choice, timing and legality vary by country.
No pesticide doses are given on purpose - always follow the product label.
"""
from __future__ import annotations

from .labels import parse_label

DISCLAIMER = (
    "General educational guidance only. Confirm the diagnosis with a local agricultural "
    "extension officer or agronomist (in India, e.g. your nearest Krishi Vigyan Kendra) before "
    "treating a crop. Always follow pesticide labels, local regulations, pre-harvest intervals "
    "and wear protective equipment."
)

CHEMICAL_NOTE = (
    "Chemical products are a last resort and must follow the label. Rotate between different "
    "modes of action to slow resistance, observe pre-harvest intervals, and check which "
    "products are approved for your crop and country."
)

SEVERITY_ORDER = ("None", "Low", "Moderate", "High", "Critical")

_SPREAD_NOTE = {
    "Low": "Usually minor; keep watching.",
    "Moderate": "Can reduce yield if ignored; act within days.",
    "High": "Can cause serious losses; act quickly.",
    "Critical": "Can destroy a crop or orchard; act immediately and alert neighbours/authorities.",
    "None": "No action needed beyond routine care.",
}

# ------------------------------------------------------------------ diseases
# plant: keyword found in the lower-cased plant name; match: keywords that must ALL appear
# in the lower-cased condition name.
_DISEASES: list[dict] = [
    {
        "plant": "apple", "match": ("scab",), "name": "Apple scab", "type": "Fungal", "severity": "Moderate",
        "agent": "Venturia inaequalis",
        "summary": "A fungal disease that leaves olive-green to brown scabby spots on leaves and fruit, "
                   "weakening the tree and spoiling the harvest.",
        "symptoms": ["Olive-green to brown velvety spots on leaves that later turn dark and corky",
                     "Yellowing, curling leaves and early leaf drop",
                     "Rough, cracked scab-like patches on fruit"],
        "favours": "Cool, wet spring weather; the fungus survives the winter in fallen leaves.",
        "immediate": ["Rake up and remove fallen leaves and infected fruit from the orchard",
                      "Prune crowded branches so air and sunlight reach the canopy",
                      "Begin a protectant spray programme at the early-leaf stage if infection is established"],
        "organic": ["Sulfur or lime-sulfur sprays applied before wet infection periods",
                    "Copper-based sprays early in the season (can scorch some varieties)",
                    "Shred or mow fallen leaves so they break down faster over winter"],
        "chemical": ["Protectant fungicides such as captan or mancozeb (where approved)",
                     "Systemic fungicides such as myclobutanil, used in rotation with other groups"],
        "prevention": ["Plant scab-resistant varieties", "Prune every year for an open canopy",
                       "Clean up leaf litter each autumn", "Avoid overhead irrigation"],
    },
    {
        "plant": "apple", "match": ("black", "rot"), "name": "Apple black rot", "type": "Fungal", "severity": "High",
        "agent": "Botryosphaeria obtusa",
        "summary": "A fungal disease causing 'frogeye' leaf spots, rotting fruit and cankers on branches.",
        "symptoms": ["Small purple spots on leaves that grow into brown 'frogeye' lesions",
                     "Brown rotting fruit that turns black and shrivels into mummies",
                     "Sunken, cracked cankers on limbs"],
        "favours": "Warm, humid weather; spores come from cankers, dead wood and mummified fruit.",
        "immediate": ["Prune out cankers and dead wood, cutting well below the damaged area, and remove it from the orchard",
                      "Pick off and destroy mummified fruit",
                      "Protect fresh pruning wounds and avoid wounding the bark"],
        "organic": ["Strict sanitation: removing every source of spores is the main control",
                    "Copper or sulfur sprays can reduce spread (follow the label)"],
        "chemical": ["Captan or thiophanate-methyl protectant sprays from bloom through early summer (where approved)"],
        "prevention": ["Remove dead wood and mummies every winter", "Keep trees vigorous and well watered",
                       "Control insects that wound fruit", "Prune in dry weather"],
    },
    {
        "plant": "apple", "match": ("rust",), "name": "Cedar apple rust", "type": "Fungal", "severity": "Moderate",
        "agent": "Gymnosporangium juniperi-virginianae",
        "summary": "A fungal disease that needs both apple and juniper (red cedar) to complete its life cycle; "
                   "it causes bright yellow-orange leaf spots.",
        "symptoms": ["Bright yellow-orange spots on the upper leaf surface, often with small black dots",
                     "Tube-like growths on the underside of spotted leaves",
                     "Spotted, deformed fruit and early leaf drop in severe cases"],
        "favours": "Wet spring weather near infected junipers/red cedars.",
        "immediate": ["Remove nearby juniper/cedar galls if practical", "Remove badly affected leaves",
                      "Apply protectant fungicide from the pink-bud stage if the disease is a yearly problem"],
        "organic": ["Remove galls from junipers in late winter", "Sulfur-based sprays before infection periods"],
        "chemical": ["Systemic fungicides such as myclobutanil, timed from pink bud through petal fall (where approved)"],
        "prevention": ["Choose rust-resistant apple varieties", "Avoid planting apples near junipers",
                       "Improve air circulation by pruning"],
    },
    {
        "plant": "cherry", "match": ("powdery",), "name": "Cherry powdery mildew", "type": "Fungal", "severity": "Moderate",
        "agent": "Podosphaera clandestina",
        "summary": "A fungal disease producing white powdery growth on leaves, shoots and sometimes fruit.",
        "symptoms": ["White, powdery patches on young leaves", "Curled, distorted or puckered leaves",
                     "White coating or russeting on fruit"],
        "favours": "Warm days, cool humid nights and dense, shaded canopies.",
        "immediate": ["Prune out infected shoots and thin the canopy", "Avoid high-nitrogen fertiliser that pushes soft growth",
                      "Start a sulfur or potassium bicarbonate spray programme"],
        "organic": ["Sulfur or potassium bicarbonate sprays", "Neem or horticultural oil sprays"],
        "chemical": ["Systemic fungicides such as myclobutanil or trifloxystrobin, rotated (where approved)"],
        "prevention": ["Prune for light and airflow", "Plant resistant varieties where available",
                       "Do not over-fertilise with nitrogen"],
    },
    {
        "plant": "corn", "match": ("gray",), "name": "Corn gray leaf spot", "type": "Fungal", "severity": "High",
        "agent": "Cercospora zeae-maydis",
        "summary": "A fungal leaf disease that creates rectangular gray-tan lesions and can cut yield sharply.",
        "symptoms": ["Narrow, rectangular tan-gray lesions that run parallel to the leaf veins",
                     "Lesions join together and the leaf dies", "Worst on lower leaves first, then moving up"],
        "favours": "Warm, humid weather, dense planting and fields with corn residue from the last crop.",
        "immediate": ["Scout the field and note how high up the plant the spots reach",
                      "Consider a fungicide at tassel stage if lesions reach the ear leaf on a susceptible hybrid",
                      "Plan residue management for after harvest"],
        "organic": ["Rotate with a non-host crop such as soybean", "Bury or break down infected residue (tillage)"],
        "chemical": ["Strobilurin or triazole fungicides at the right growth stage (where approved)"],
        "prevention": ["Grow resistant hybrids", "Rotate crops for at least a year", "Avoid very dense planting",
                       "Manage residue after harvest"],
    },
    {
        "plant": "corn", "match": ("rust",), "name": "Corn common rust", "type": "Fungal", "severity": "Moderate",
        "agent": "Puccinia sorghi",
        "summary": "A fungal disease that produces cinnamon-brown pustules on both sides of the leaf.",
        "symptoms": ["Small, round to oval, powdery brown-red pustules on both leaf surfaces",
                     "Pustules turn black late in the season", "Heavy infection makes leaves yellow and dry"],
        "favours": "Cool, humid weather with heavy dew; wind carries spores long distances.",
        "immediate": ["Scout fields weekly once pustules appear", "Spray only if infection is heavy on young plants and the hybrid is susceptible"],
        "organic": ["Plant resistant hybrids", "Early planting can help the crop outgrow the disease"],
        "chemical": ["Triazole or strobilurin fungicides if infection is severe early (where approved)"],
        "prevention": ["Use resistant hybrids", "Avoid planting very late", "Monitor regional rust reports"],
    },
    {
        "plant": "corn", "match": ("northern",), "name": "Northern corn leaf blight", "type": "Fungal", "severity": "High",
        "agent": "Exserohilum turcicum",
        "summary": "A fungal disease causing long, cigar-shaped gray-green lesions that reduce photosynthesis and yield.",
        "symptoms": ["Long (2-15 cm) cigar-shaped gray-green or tan lesions", "Lesions merge and leaves look scorched",
                     "Dark spore growth on lesions in humid weather"],
        "favours": "Moderate temperatures, heavy dew and long leaf-wetness periods; survives on crop residue.",
        "immediate": ["Scout fields and check how far up the plant lesions reach", "Apply fungicide near tassel if disease is spreading on susceptible hybrids",
                      "Plan crop rotation and residue management"],
        "organic": ["Rotate crops", "Till or bury infected residue"],
        "chemical": ["Strobilurin or triazole fungicides at tassel stage (where approved)"],
        "prevention": ["Grow resistant hybrids", "Rotate with non-host crops", "Manage residue after harvest"],
    },
    {
        "plant": "grape", "match": ("black", "rot"), "name": "Grape black rot", "type": "Fungal", "severity": "High",
        "agent": "Guignardia bidwellii",
        "summary": "A fungal disease that spots leaves and turns berries into hard black mummies.",
        "symptoms": ["Tan leaf spots with dark borders and tiny black dots", "Berries turn brown, then shrivel into hard black mummies",
                     "Dark lesions on shoots and tendrils"],
        "favours": "Warm, wet weather from shoot growth through a few weeks after bloom.",
        "immediate": ["Remove and destroy mummified berries and infected leaves", "Open the canopy by shoot thinning and leaf removal",
                      "Apply protectant fungicide from early shoot growth to about four weeks after bloom"],
        "organic": ["Sanitation is the main tool: remove all mummies from vines and the ground", "Copper or sulfur sprays can help in low-pressure years"],
        "chemical": ["Mancozeb, captan or myclobutanil-type fungicides on a protective schedule (where approved)"],
        "prevention": ["Prune and train vines for airflow", "Remove mummies every winter", "Control weeds under the vines"],
    },
    {
        "plant": "grape", "match": ("esca",), "name": "Grape esca (black measles)", "type": "Fungal", "severity": "High",
        "agent": "A complex of wood-rotting fungi",
        "summary": "A trunk disease with no cure that causes tiger-stripe leaves, spotted berries and slow vine decline.",
        "symptoms": ["Yellow or red 'tiger-stripe' patterns between leaf veins", "Dark spots on berries ('measles') and shriveled fruit",
                     "Dead wood or dark streaks inside the trunk; sudden vine collapse in summer"],
        "favours": "Infection through pruning wounds on older vines; worse in stressed vines.",
        "immediate": ["Mark affected vines and avoid spreading the disease with pruning tools", "Cut out infected wood and burn it",
                      "Disinfect pruning tools between vines and seal large pruning wounds"],
        "organic": ["Remove and destroy dead or declining vines", "Trunk renewal: cut back to healthy wood below the infection"],
        "chemical": ["No fully effective chemical cure; protectant wound treatments may reduce new infections"],
        "prevention": ["Prune in dry weather, late in the dormant season", "Make small clean pruning cuts and protect wounds",
                       "Use clean planting material", "Reduce vine stress (water, nutrition)"],
    },
    {
        "plant": "grape", "match": ("blight",), "name": "Grape leaf blight (Isariopsis leaf spot)", "type": "Fungal", "severity": "Moderate",
        "agent": "Pseudocercospora vitis (Isariopsis)",
        "summary": "A fungal leaf spot that causes dark irregular patches and early leaf fall, weakening the vine.",
        "symptoms": ["Irregular dark brown to black spots, often with a yellow edge", "Spots merge and the leaf dries out",
                     "Premature leaf drop late in the season"],
        "favours": "Warm, humid weather and dense canopies.",
        "immediate": ["Remove badly spotted leaves", "Thin the canopy for airflow", "Spray a protectant fungicide if spreading"],
        "organic": ["Copper-based sprays (follow the label)", "Remove and destroy fallen leaves"],
        "chemical": ["Mancozeb or copper-based fungicides (where approved)"],
        "prevention": ["Prune and train for open canopies", "Avoid overhead irrigation", "Clean up fallen leaves"],
    },
    {
        "plant": "orange", "match": ("greening",), "name": "Citrus greening (Huanglongbing)", "type": "Bacterial", "severity": "Critical",
        "agent": "Candidatus Liberibacter asiaticus, spread by the Asian citrus psyllid",
        "summary": "A deadly bacterial disease with no cure. It makes fruit small, lopsided and bitter and slowly kills the tree.",
        "symptoms": ["Blotchy, asymmetrical yellow mottling of leaves", "Small, lopsided, green-bottomed, bitter fruit",
                     "Twig dieback and general decline"],
        "favours": "Spread by psyllid insects and infected planting material.",
        "immediate": ["Report suspected trees to your agriculture department - it is a notifiable disease in many regions",
                      "Remove and destroy confirmed infected trees as advised locally", "Start controlling psyllids now"],
        "organic": ["Remove infected trees quickly to protect healthy ones", "Release or protect natural enemies of the psyllid"],
        "chemical": ["Insecticide programmes against the psyllid vector, coordinated with neighbouring growers"],
        "prevention": ["Plant only certified disease-free nursery trees", "Monitor with yellow sticky traps",
                       "Coordinate psyllid control across the area", "Do not move citrus cuttings between regions"],
    },
    {
        "plant": "peach", "match": ("bacterial",), "name": "Peach bacterial spot", "type": "Bacterial", "severity": "High",
        "agent": "Xanthomonas arboricola pv. pruni",
        "summary": "A bacterial disease that causes angular leaf spots, shot-holes and blemished, cracked fruit.",
        "symptoms": ["Small, angular, water-soaked spots that turn purple-brown", "Spots drop out leaving a 'shot-hole' look and leaves fall early",
                     "Pitted, cracked spots on fruit"],
        "favours": "Warm, wet, windy weather; bacteria overwinter in twig lesions.",
        "immediate": ["Prune out and remove infected twigs in dry weather", "Avoid overhead irrigation",
                      "Apply copper or an approved bactericide following local advice"],
        "organic": ["Copper sprays at leaf fall and early spring (can harm leaves if overused)", "Balanced nitrogen - avoid excess"],
        "chemical": ["Oxytetracycline-based products in some regions, applied per label and local advice"],
        "prevention": ["Plant tolerant varieties", "Keep trees healthy and wind-protected", "Prune for airflow",
                       "Avoid excess nitrogen"],
    },
    {
        "plant": "pepper", "match": ("bacterial",), "name": "Pepper bacterial spot", "type": "Bacterial", "severity": "High",
        "agent": "Xanthomonas species",
        "summary": "A bacterial disease that spots leaves and fruit, causing defoliation and unmarketable peppers.",
        "symptoms": ["Small water-soaked spots turning brown with a yellow halo", "Leaves yellow and drop, exposing fruit to sunscald",
                     "Raised, scabby spots on fruit"],
        "favours": "Warm, wet weather; spread by splashing water, tools and infected seed.",
        "immediate": ["Remove and destroy badly infected plants and debris", "Stop overhead watering; do not work among wet plants",
                      "Apply a copper-based bactericide where advised"],
        "organic": ["Copper sprays (limited effect - best used early)", "Rotate away from peppers and tomatoes for 2-3 years"],
        "chemical": ["Copper plus mancozeb tank mixes are commonly recommended where approved"],
        "prevention": ["Use disease-free seed and transplants", "Rotate crops", "Drip irrigation instead of sprinklers",
                       "Disinfect tools and stakes"],
    },
    {
        "plant": "potato", "match": ("early", "blight"), "name": "Potato early blight", "type": "Fungal", "severity": "Moderate",
        "agent": "Alternaria solani",
        "summary": "A fungal disease that creates target-like brown spots on older leaves and reduces yield when plants are stressed.",
        "symptoms": ["Dark brown spots with concentric 'bullseye' rings on older leaves", "Yellow tissue around the spots",
                     "Leaves dry up and drop; dark sunken spots on tubers in bad cases"],
        "favours": "Warm weather with alternating wet and dry periods, and plants stressed by low nitrogen or drought.",
        "immediate": ["Remove the worst infected lower leaves", "Water at the base and avoid wetting the foliage",
                      "Start a protectant fungicide programme if the disease is spreading"],
        "organic": ["Copper-based fungicides (follow the label)", "Mulch to stop soil splash", "Keep plants well fed and watered"],
        "chemical": ["Chlorothalonil, mancozeb or azoxystrobin (where approved), rotated"],
        "prevention": ["Rotate crops (2-3 years)", "Use certified seed and resistant varieties", "Avoid plant stress",
                       "Destroy crop debris after harvest"],
    },
    {
        "plant": "potato", "match": ("late", "blight"), "name": "Potato late blight", "type": "Fungal", "severity": "Critical",
        "agent": "Phytophthora infestans (an oomycete, fungus-like organism)",
        "summary": "A fast-spreading disease that can destroy a field in days in cool wet weather. It was behind the Irish Potato Famine.",
        "symptoms": ["Dark, water-soaked blotches on leaves that quickly turn brown-black", "White fuzzy mold on the underside in humid weather",
                     "Brown rot on stems; firm brown-red rot in tubers"],
        "favours": "Cool (10-20 C), humid and rainy weather; spreads by wind and rain from infected plants and tubers.",
        "immediate": ["Remove and bag infected plants - do not compost them", "Spray a protectant fungicide on healthy plants straight away",
                      "Avoid overhead irrigation and do not harvest wet", "Warn neighbouring growers"],
        "organic": ["Copper-based sprays as a protectant (they do not cure existing infection)", "Hill soil well to protect tubers",
                    "Destroy infected vines and let tubers cure before harvest"],
        "chemical": ["Protectants such as mancozeb or chlorothalonil", "Systemic products (for example cymoxanil or metalaxyl-based mixes) where approved and rotated"],
        "prevention": ["Plant certified disease-free seed potatoes", "Choose resistant varieties", "Destroy volunteer potatoes and cull piles",
                       "Space plants for airflow and water at the base"],
    },
    {
        "plant": "squash", "match": ("powdery",), "name": "Squash powdery mildew", "type": "Fungal", "severity": "Moderate",
        "agent": "Podosphaera xanthii",
        "summary": "A fungal disease with white powdery patches on leaves that reduces yield and fruit quality.",
        "symptoms": ["White powdery spots on upper leaf surfaces and stems", "Leaves yellow, curl and die",
                     "Smaller, less sweet fruit"],
        "favours": "Warm, dry days with humid nights; shaded, crowded plants.",
        "immediate": ["Remove heavily infected leaves", "Thin vines and improve airflow", "Spray sulfur or potassium bicarbonate"],
        "organic": ["Sulfur or potassium bicarbonate sprays", "Neem or horticultural oil", "Water in the morning at the base of the plant"],
        "chemical": ["Systemic fungicides such as myclobutanil or strobilurins in rotation (where approved)"],
        "prevention": ["Plant resistant varieties", "Give plants enough space", "Avoid excess nitrogen"],
    },
    {
        "plant": "strawberry", "match": ("scorch",), "name": "Strawberry leaf scorch", "type": "Fungal", "severity": "Moderate",
        "agent": "Diplocarpon earlianum",
        "summary": "A fungal leaf disease that makes leaves look scorched and weakens plants over time.",
        "symptoms": ["Many small, irregular purple spots on the upper leaf", "Spots merge and the leaf turns brown and scorched",
                     "Leaves curl and die in severe cases"],
        "favours": "Warm, humid weather and wet foliage; spreads from old infected leaves.",
        "immediate": ["Remove infected leaves", "Avoid overhead irrigation",
                      "Renovate beds after harvest: mow or trim old foliage and remove it"],
        "organic": ["Remove and destroy old leaves", "Copper sprays in early spring (follow the label)", "Keep rows narrow and weed-free for airflow"],
        "chemical": ["Captan or myclobutanil-type fungicides in severe cases (where approved)"],
        "prevention": ["Plant resistant varieties and clean plants", "Space plants well", "Renovate beds yearly",
                       "Water at the base"],
    },
    {
        "plant": "tomato", "match": ("bacterial",), "name": "Tomato bacterial spot", "type": "Bacterial", "severity": "High",
        "agent": "Xanthomonas species",
        "summary": "A bacterial disease that spots leaves and fruit and spreads quickly in warm, wet weather.",
        "symptoms": ["Small dark, greasy-looking spots with yellow halos on leaves", "Spots can tear, leaving ragged holes",
                     "Raised scabby spots on fruit"],
        "favours": "Warm, wet weather; spread by rain splash, tools and infected seed or transplants.",
        "immediate": ["Remove and destroy badly infected plants", "Stop overhead watering and avoid handling wet plants",
                      "Apply a copper-based bactericide where advised"],
        "organic": ["Copper sprays early and regularly (limited effect on severe cases)", "Rotate away from tomatoes and peppers for 2-3 years"],
        "chemical": ["Copper plus mancozeb tank mixes are commonly recommended where approved"],
        "prevention": ["Use clean seed and transplants", "Drip irrigation", "Disinfect tools and stakes",
                       "Remove crop debris after harvest"],
    },
    {
        "plant": "tomato", "match": ("early", "blight"), "name": "Tomato early blight", "type": "Fungal", "severity": "Moderate",
        "agent": "Alternaria solani",
        "summary": "A common fungal disease with target-like spots on older leaves; it weakens plants and exposes fruit to sunscald.",
        "symptoms": ["Brown spots with concentric rings on older (lower) leaves", "Yellowing around the spots and leaf drop",
                     "Dark sunken lesions near the fruit stem"],
        "favours": "Warm, humid weather with alternating wet and dry periods; stressed plants.",
        "immediate": ["Remove infected lower leaves and bag them", "Mulch the soil and water at the base",
                      "Spray a protectant fungicide if spreading"],
        "organic": ["Copper or Bacillus-based biofungicides", "Stake or cage plants to keep leaves off the soil", "Mulch to prevent soil splash"],
        "chemical": ["Chlorothalonil, mancozeb or azoxystrobin (where approved), rotated"],
        "prevention": ["Rotate crops 2-3 years", "Use resistant varieties", "Space plants for airflow",
                       "Keep plants well fed and watered"],
    },
    {
        "plant": "tomato", "match": ("late", "blight"), "name": "Tomato late blight", "type": "Fungal", "severity": "Critical",
        "agent": "Phytophthora infestans (an oomycete, fungus-like organism)",
        "summary": "A rapid, destructive disease that can wipe out tomatoes in days during cool, wet weather.",
        "symptoms": ["Large, greasy, gray-green to dark brown blotches on leaves and stems", "White fuzzy growth under leaves in humid weather",
                     "Firm brown, greasy-looking rot on fruit"],
        "favours": "Cool, wet, humid weather; spreads by wind and rain, also from infected potatoes nearby.",
        "immediate": ["Remove and bag infected plants - do not compost them", "Spray a protectant fungicide on healthy plants immediately",
                      "Stop overhead watering", "Check nearby potatoes and warn neighbours"],
        "organic": ["Copper-based sprays as a protectant before infection", "Remove infected plants early to protect others"],
        "chemical": ["Protectants such as chlorothalonil or mancozeb", "Systemic products where approved, rotated by mode of action"],
        "prevention": ["Choose resistant varieties", "Space and stake plants for airflow", "Water at the base in the morning",
                       "Destroy volunteer tomatoes and potatoes"],
    },
    {
        "plant": "tomato", "match": ("mold",), "name": "Tomato leaf mold", "type": "Fungal", "severity": "Moderate",
        "agent": "Passalora fulva",
        "summary": "A fungal disease of humid places, especially greenhouses, with pale spots above and velvety mold below the leaf.",
        "symptoms": ["Pale green or yellow spots on the upper leaf surface", "Olive-green to brown velvety mold on the underside",
                     "Leaves curl, wither and drop"],
        "favours": "High humidity (above 85%) and poor ventilation.",
        "immediate": ["Increase ventilation and lower humidity", "Remove infected leaves and bag them",
                      "Water at the base, early in the day"],
        "organic": ["Prune lower leaves for airflow", "Copper or Bacillus-based biofungicides"],
        "chemical": ["Chlorothalonil or copper fungicides where approved"],
        "prevention": ["Grow resistant varieties", "Space plants and ventilate greenhouses", "Avoid wetting leaves",
                       "Clean the greenhouse between crops"],
    },
    {
        "plant": "tomato", "match": ("septoria",), "name": "Tomato Septoria leaf spot", "type": "Fungal", "severity": "Moderate",
        "agent": "Septoria lycopersici",
        "summary": "A fungal leaf disease causing many small spots; it can strip a plant of leaves from the bottom up.",
        "symptoms": ["Many small round spots with dark borders and gray-white centres", "Tiny black dots in the middle of spots",
                     "Lower leaves yellow and fall off first"],
        "favours": "Warm, wet weather and splashing water; fungus lives on crop debris.",
        "immediate": ["Remove infected lower leaves", "Mulch under plants to stop soil splash", "Spray a protectant fungicide if spreading"],
        "organic": ["Copper or Bacillus-based sprays", "Stake plants and prune lower leaves", "Clear debris at the end of the season"],
        "chemical": ["Chlorothalonil, mancozeb or copper fungicides (where approved)"],
        "prevention": ["Rotate crops 2-3 years", "Water at the base", "Use mulch", "Remove infected debris"],
    },
    {
        "plant": "tomato", "match": ("spider",), "name": "Tomato spider mites", "type": "Pest", "severity": "Moderate",
        "agent": "Two-spotted spider mite (Tetranychus urticae)",
        "summary": "Tiny sap-sucking mites that cause stippled, yellowing leaves and fine webbing, especially in hot dry weather.",
        "symptoms": ["Tiny pale speckles (stippling) on leaves", "Leaves turn yellow or bronze and dry out",
                     "Fine webbing on the underside of leaves and between stems"],
        "favours": "Hot, dry, dusty conditions; drought-stressed plants.",
        "immediate": ["Spray plants with a strong stream of water, underside of leaves included", "Apply insecticidal soap or neem oil, repeating as directed",
                      "Remove heavily infested leaves"],
        "organic": ["Insecticidal soap, neem or horticultural oil", "Release predatory mites (for example Phytoseiulus persimilis)",
                    "Keep plants watered and reduce dust"],
        "chemical": ["Miticides only if needed, chosen to spare beneficial insects (follow the label)"],
        "prevention": ["Inspect leaf undersides regularly", "Avoid broad-spectrum insecticides that kill natural enemies",
                       "Keep plants well watered", "Remove weeds nearby"],
    },
    {
        "plant": "tomato", "match": ("target",), "name": "Tomato target spot", "type": "Fungal", "severity": "Moderate",
        "agent": "Corynespora cassiicola",
        "summary": "A fungal disease causing brown ringed spots on leaves, stems and fruit.",
        "symptoms": ["Brown spots with concentric rings and a yellow halo", "Spots merge, leaves yellow and fall",
                     "Sunken dark spots on fruit"],
        "favours": "Warm, humid weather and long leaf-wetness periods in dense canopies.",
        "immediate": ["Remove infected leaves", "Improve airflow by pruning and spacing", "Apply a protectant fungicide if spreading"],
        "organic": ["Copper or Bacillus-based sprays", "Mulch and stake plants"],
        "chemical": ["Chlorothalonil or azoxystrobin-type fungicides (where approved)"],
        "prevention": ["Rotate crops", "Avoid overhead irrigation", "Clean up crop debris", "Space plants well"],
    },
    {
        "plant": "tomato", "match": ("yellow",), "name": "Tomato yellow leaf curl virus", "type": "Viral", "severity": "Critical",
        "agent": "Tomato yellow leaf curl virus, spread by whiteflies (Bemisia tabaci)",
        "summary": "A viral disease with no cure. Plants stay small, leaves curl up and yellow, and fruit set collapses.",
        "symptoms": ["Leaves curl upward and turn yellow at the edges", "Stunted, bushy plants",
                     "Flowers drop and very little fruit forms"],
        "favours": "Warm weather with plenty of whiteflies; infected plants and weeds act as virus sources.",
        "immediate": ["Remove and destroy infected plants (bag them)", "Control whiteflies: yellow sticky traps, insecticidal soap or neem oil",
                      "Remove weeds that host whiteflies"],
        "organic": ["Reflective mulch to repel whiteflies", "Insect-proof nets over seedlings and nurseries", "Insecticidal soap or neem for whitefly control"],
        "chemical": ["Insecticides against whiteflies where approved, rotated; they protect healthy plants but do not cure infected ones"],
        "prevention": ["Grow resistant or tolerant varieties", "Use virus-free transplants", "Install insect nets in nurseries",
                       "Do not grow tomatoes next to infected crops"],
    },
    {
        "plant": "tomato", "match": ("mosaic",), "name": "Tomato mosaic virus", "type": "Viral", "severity": "High",
        "agent": "Tomato mosaic virus (ToMV), a tobamovirus",
        "summary": "A viral disease with no cure that causes mottled, distorted leaves and poor fruit. It spreads by touch and tools.",
        "symptoms": ["Light and dark green mosaic mottling on leaves", "Curled, narrow or fern-like leaves",
                     "Stunted growth and uneven fruit ripening"],
        "favours": "Spread mechanically by hands, tools and contaminated seed; survives on debris.",
        "immediate": ["Remove and destroy infected plants (do not compost)", "Wash hands and disinfect tools between plants",
                      "Do not handle healthy plants after touching infected ones"],
        "organic": ["Strict hygiene is the only control", "Remove weeds that can host the virus"],
        "chemical": ["No chemical cure exists; insecticides do not help"],
        "prevention": ["Use resistant varieties and certified seed", "Disinfect tools and stakes", "Wash hands, especially after handling tobacco",
                       "Rotate crops and clear debris"],
    },
]

# ------------------------------------------------------------------ healthy plants
_HEALTHY_TIPS: dict[str, str] = {
    "apple": "Prune in late winter for an open canopy and clear fallen leaves each autumn to prevent scab.",
    "blueberry": "Keep soil acidic and mulched, and water regularly - blueberries have shallow roots.",
    "cherry": "Prune for airflow and avoid heavy nitrogen to reduce powdery mildew risk.",
    "corn": "Rotate crops and manage residue after harvest to reduce leaf-blight pressure.",
    "grape": "Open the canopy with shoot thinning and remove mummified berries every winter.",
    "peach": "Prune for light and airflow, and avoid overhead irrigation to reduce bacterial spot.",
    "pepper": "Use drip irrigation and rotate away from peppers and tomatoes to avoid bacterial spot.",
    "potato": "Use certified seed potatoes, hill soil around plants and scout for late blight in cool, wet weather.",
    "raspberry": "Thin canes for airflow and remove old fruited canes after harvest.",
    "soybean": "Rotate with a non-legume crop and scout fields regularly for leaf spots and pests.",
    "strawberry": "Renovate beds after harvest and remove old leaves to prevent leaf diseases.",
    "tomato": "Stake plants, mulch the soil and water at the base to keep foliage dry.",
}

_GENERAL_HEALTHY_STEPS = [
    "Keep inspecting leaves weekly, especially the undersides",
    "Water at the base of the plant, early in the day",
    "Feed with balanced nutrients; avoid too much nitrogen",
    "Remove fallen leaves and plant debris regularly",
]


def _fill(entry: dict, plant_label: str) -> dict:
    out = {
        "known": True, "healthy": False, "plant_label": plant_label,
        "agent": "", "favours": "", "symptoms": [], "immediate": [], "organic": [],
        "chemical": [], "prevention": [],
    }
    out.update(entry)
    out["spread_note"] = _SPREAD_NOTE.get(out.get("severity", "Moderate"), "")
    return out


def _healthy_info(plant_label: str) -> dict:
    key = next((k for k in _HEALTHY_TIPS if k in plant_label.lower()), None)
    tip = _HEALTHY_TIPS.get(key, "Keep following good crop-care practices.")
    return _fill({
        "name": f"Healthy {plant_label.lower()}", "type": "Healthy", "severity": "None", "healthy": True,
        "summary": "No disease signs were detected. The leaf looks healthy to the model.",
        "symptoms": ["Even green colour with no spots, mold or distortion"],
        "immediate": _GENERAL_HEALTHY_STEPS[:2],
        "prevention": [tip] + _GENERAL_HEALTHY_STEPS[2:],
    }, plant_label)


def _generic_info(plant_label: str, condition: str) -> dict:
    return _fill({
        "name": f"{plant_label} - {condition}", "type": "Other", "severity": "Moderate", "known": False,
        "summary": "No detailed guide is stored for this class yet.",
        "symptoms": [],
        "immediate": ["Isolate the affected plant if possible", "Take clear photos and ask a local agronomist to confirm",
                      "Avoid spreading the problem on tools or hands"],
        "prevention": ["Rotate crops", "Water at the base", "Remove plant debris"],
    }, plant_label)


def get_info(class_name: str) -> dict:
    """Return the knowledge-base entry for a PlantVillage class folder name."""
    plant, condition, healthy = parse_label(class_name)
    if healthy:
        return _healthy_info(plant)
    p, c = plant.lower(), condition.lower()
    for entry in _DISEASES:
        if entry["plant"] in p and all(k in c for k in entry["match"]):
            return _fill(entry, plant)
    return _generic_info(plant, condition)


DISEASE_TYPES = ("Fungal", "Bacterial", "Viral", "Pest", "Healthy", "Other")
