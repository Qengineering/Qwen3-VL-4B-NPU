# Qwen3-VL-4B NPU
![Alt text](https://github.com/user-attachments/assets/6d297a34-c516-4cb1-be4a-bca471d40fa6)
<br><br>**User**:\<image\>Describe the image.<br><br>
**Answer**: This is a surreal, stylized digital illustration depicting an astronaut relaxing on the Moon.
Here's a breakdown of the scene:
- **The Astronaut**: A space traveler in a white spacesuit reclines on their back on the lunar surface. Their helmet visor reflects the surrounding environment with a golden, metallic sheen. They are holding a green glass bottle—likely a beer—with both hands, as if taking a sip or savoring a moment of leisure.
- **The Setting**: The astronaut is lying on a small, green cooler that has some white lettering (possibly "Coca-Cola" or similar branding) on its side. To the right, a metal ladder leans against the ground, suggesting this might be part of an expedition or base camp.
- **Background**: In the distance, Earth looms large in the sky, its blue and green surface visible against the blackness of space. The Moon's surface is depicted with craters and rocks, typical of lunar terrain.
- **Atmosphere**: The image has a dreamlike, almost cinematic quality. It juxtaposes the harsh reality of space exploration with a moment of casual relaxation—a humorous or ironic twist on the idea of astronauts enjoying a drink in such an alien environment.

Overall, it’s a visually striking and imaginative piece that blends science fiction with everyday leisure, creating a whimsical and slightly absurd scene.

------------

## Qwen3-4B VLM for RK3588 NPU (Rock 5, Orange Pi 5). <br/>
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)<br/><br/>
Paper: [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388)<br/><br/>
Hugging face: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

------------

## Introduction

LLMs (Large Language Models) are neural networks trained on large text datasets to understand and generate language.<br>
VLMs (Vision-Language Models) add a visual encoder so the model can process images and text together.<br> 
A combined VLM+LLM system is often referred to as a multimodal model.

These models can be large—hundreds of millions to billions of parameters—which impacts accuracy, memory use, and runtime speed.<br>
On edge devices like the RK3588, available RAM and compute are limited, and even the NPU has strict constraints on supported operations.<br>
Because of this, models typically need to be quantised or simplified to fit.

Performance is usually expressed in tokens (words) per second.<br>
Once converted to RKNN, parts of the model can run on the NPU, improving speed.<br>
Despite these limits, models like Qwen3 run well on the RK3588 because the NPU efficiently accelerates the heavy math, and the vision encoder can be optimised. This makes advanced multimodal AI feasible on small, power-efficient devices.

------------

## Model performance benchmark (FPS)

All models, with C++ examples, can be found on the Q-engineering GitHub.<br><br>
All LLM models are quantized to **w8a8**, while the VLM vision encoders use **fp16**.<br>

| model         | RAM (GB)<sup>1</sup> | llm cold sec<sup>2</sup> | llm warm sec<sup>3</sup> | vlm cold sec<sup>2</sup> | vlm warm sec<sup>3</sup> | Resolution | Tokens/s |
| --------------| :--: | :-----: | :-----: | :--------: | :-----: | :--------:  | :--------: |
| [Qwen3-2B](https://github.com/Qengineering/Qwen3-VL-2B-NPU) | 3.1 | 21.9 | 2.6 | 10.0  | 0.9 | 448 x 448 | 11.5 |
| [Qwen3-4B](https://github.com/Qengineering/Qwen3-VL-4B-NPU) | 8.7 | 49.6 | 5.6 | 10.6  | 1.1 | 448 x 448 | 5.7 |
| [Qwen2.5-3B](https://github.com/Qengineering/Qwen2.5-VL-3B-NPU) | 4.8 | 48.3 |  4.0 | 17.9  | 1.8 | 392 x 392 | 7.0 |
| [Qwen2-7B](https://github.com/Qengineering/Qwen2-VL-7B-NPU) | 8.7 | 86.6 |   34.5 | 37.1  | 20.7 | 392 x 392 | 3.7 |
| [Qwen2-2.2B](https://github.com/Qengineering/Qwen2-VL-2B-NPU) | 3.3 | 29.1 |   2.5 | 17.1  | 1.7 | 392 x 392 | 12.5 |
| [InternVL3-1B](https://github.com/Qengineering/InternVL3-NPU) | 1.3 |  6.8 |   1.1 | 7.8    | 0.75 | 448 x 448 | 30 |
| [SmolVLM2-2.2B](https://github.com/Qengineering/SmolVLM2-2B-NPU) | 3.4 | 21.2 |   2.6 | 10.5   | 0.9  | 384 x 384 | 11 |
| [SmolVLM2-500M](https://github.com/Qengineering/SmolVLM2-500M-NPU) | 0.8 |  4.8 |   0.7 | 2.5    | 0.25 | 384 x 384 | 31 |
| [SmolVLM2-256M](https://github.com/Qengineering/SmolVLM2-256M-NPU) | 0.5 |  1.1 |   0.4 | 2.5    | 0.25 | 384 x 384 | 54 |

<sup>1</sup> The total used memory; LLM plus the VLM. <br>
<sup>2</sup> When an llm/vlm model is loaded for the first time from your disk to RAM or NPU, it is called a cold start.<br>
The duration depends on your OS, I/O transfer rate, and memory mapping.<br> 
<sup>3</sup> Subsequent loading (warm start) takes advantage of the already mapped data in RAM. Mostly, only a few pointers need to be restored.<br><br>
<img width="600" height="450" alt="Plot_1" src="https://github.com/user-attachments/assets/6dbb7d8a-c55b-49b4-ae5a-ac272aade60b" /><br>
<img width="600" height="450" alt="Plot_2" src="https://github.com/user-attachments/assets/c4dd409b-2533-4ae0-942c-58fb3e8f5834" />

------------

## Dependencies.
To run the application, you have to:
- OpenCV 64-bit installed.
- rkllm library.
- rknn library.
- Optional: Code::Blocks. (```$ sudo apt-get install codeblocks```)

### Installing the dependencies.
Start with the usual 
```
$ sudo apt-get update 
$ sudo apt-get upgrade
$ sudo apt-get install cmake wget curl
```
#### OpenCV
To install OpenCV on your SBC, follow the Raspberry Pi 4 [guide](https://qengineering.eu/install-opencv-on-raspberry-64-os.html).<br><br>
Or, when you have no intentions to program code:
```
$ sudo apt-get install libopencv-dev 
```
------------

## Installing the app.
```
$ git clone https://github.com/Qengineering/Qwen3-VL-4B-NPU.git
```

#### RKLLM, RKNN
To run InternVL3, you need to have the **rkllm-runtime** library version **1.2.3** (or higher) installed, as well as the **rknpu driver** version **0.9.8**.<br>
If you don't have these on your machine, or if you have a lower version, you need to install them.<br>
We have provided the correct versions in the repo.<br>
```bash
$ cd ./Qwen3-VL-4B-NPU/aarch64/library
$ sudo cp ./*.so /usr/local/lib
$ cd ./Qwen3-VL-4B-NPU/aarch64/include
$ sudo cp ./*.h /usr/local/include
```
### Download the LLM and VLM model.
The next step is downloading the models.<br>
This time, we used the original model supplied by Rockchips [rkllm_model_zoo](https://console.box.lenovo.com/l/l0tXb8) (44 GB!), fetch code: rkllm.<br><br>
Better to download the two needed files (5.4 GB) from our Sync.com server:<br>
[qwen3-vl-4b-instruct_w8a8_rk3588.rkllm](https://ln5.sync.com/dl/59b6ffd20#ua4vsgkx-yx2p8ick-etf2t9tn-i5hgbkca) and [qwen3-vl-4b-vision_rk3588.rknn](https://ln5.sync.com/dl/59a88eba0#3cmd6hek-nc42fahz-765d7a4r-442xvqxi)<br>
Copy both to your `./model` folder.

### Building the app.
Once you have the two models, it is time to build your application.<br>
You can use **Code::Blocks**.
- Load the project file *.cbp in Code::Blocks.
- Select _Release_, not Debug.
- Compile and run with F9.
- You can alter command line arguments with _Project -> Set programs arguments..._ 

Or use **Cmake**.
```
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
```
### Running the app.
The app has the following arguments.
```bash
VLM_NPU Picture RKNN_model RKLLM_model NewTokens ContextLength
```
| Argument   | Comment |
| --------------| --  |
| picture | The image. Provide a dummy if you don't want to use an image | 
| RKNN_model | The visual encoder model (vlm) | 
| RKLLM_model | The large language model (llm) | 
| NewTokens | This sets the maximum number of new tokens. Optional, default 2048| 
| ContextLength | This specifies the maximum total number of tokens the model can process. Optional, default 4096| 

<br>In the context of the Rockchip RK3588 LLM (Large Language Model) library, the parameters NewTokens and ContextLength both control different limits for text generation, and they're typical in LLM workflows.<br>
**NewTokens**<br> 
This sets the maximum number of tokens (pieces of text, typically sub-word units) that the model is allowed to generate in response to a prompt during a single inference round. For example, if set to 300, the model will not return more than 300 tokens as output, regardless of the prompt length. It's important for controlling generation length to avoid too-short or too-long responses, helping manage resource use and output size.<br>
**ContextLength**<br>
This specifies the maximum total number of tokens the model can process in one go, which includes both the prompt (input) tokens and all generated tokens. For example, if set to 2048 and your prompt already uses 500 tokens, the model can generate up to 2048-500 = 1548 new tokens. This is a hardware and architecture constraint set during model conversion and deployment, as the context window cannot exceed the model's design limit (for instance, 4096 or 8192 tokens depending on the model variant).

A typical command line can be:
```bash
VLM_NPU ./Moon.jpg ./models/qwen3-vl-4b-vision_rk3588.rknn ./models/qwen3-vl-4b-instruct_w8a8_rk3588.rkllm 2048 4096
```
The NewTokens (2048) and ContextLength (4096) are optional and can be omitted.
### Using the app.
Using the application is simple. Once you provide the image and the models, you can ask everything you want.<br>
Remember, we are on a bare Rock5C, so don't expect the same quality answers as ChatGPT can provide.<br>
On the other hand, when you see the examples below, the app performs amazingly well.<br><br>
If you want to talk about the picture, you need to include the token `<image>` in your prompt once.<br>
The app remembers the dialogue until you give the token `<clear>`.<br>
With `<exit>`, you leave the application.
### C++ code.  
Below, you find the surprisingly little code of main.cpp. 
```cpp
#include "RK35llm.h"

int main(int argc, char** argv)
{
    std::string input_str;
    std::string output_str;
    RK35llm RKLLM;

    RKLLM.SetInfo(true);            //yes, you may give me additional model information
    RKLLM.SetSilence(false);        //you may print the incremental text chunks on the terminal

    if     (argc< 4) {std::cerr << "Usage: " << argv[0] << " image vlm_model llm_model [option]NewTokens [option]ContextLength\n"; return -1;}
    else if(argc==4) RKLLM.LoadModel(argv[2],argv[3]);
    else if(argc==5) RKLLM.LoadModel(argv[2],argv[3],std::atoi(argv[4]));
    else if(argc> 5) RKLLM.LoadModel(argv[2],argv[3],std::atoi(argv[4]),std::atoi(argv[5]));

    cv::Mat Pic = cv::imread(argv[1]);
    RKLLM.LoadImage(Pic);

    while(true) {
        printf("\n");
        printf("User: ");

        std::getline(std::cin, input_str);
        if (input_str == "exit") break;

        output_str = RKLLM.Ask(input_str);
//        std::cout << "\nLLM Reply: " << output_str << std::endl;
    }

    return 0;
}
```
Most code speaks for itself. One remark.<br>
The LLM generates the answer to your request in little pieces of text. You see them as if the LLM is typing on your terminal.<br>
RKLLM.SetSilence controls this behaviour. When set to `true`, it will suppress this output. RKLLM.Ask() also returns the complete answer.<br>
In main.cpp uncomment this line:
```cpp
std::cout << "\nLLM Answer: " << output_str << std::endl;
```

------------

## VLM examples.
| Image         | Dialog |
| --------------| --  |
| ![Singapore_S](https://github.com/user-attachments/assets/e742059a-3bdb-41be-9647-e7767da12ce6) |**User**: \<image\>Which city is this?<br>**Answer**: This is **Singapore**.The image shows the iconic skyline of Singapore during sunset or twilight, featuring:<br><br>- The **Marina Bay Sands** hotel on the left — with its distinctive three-tower design and rooftop pool.<br>- The **Singapore Flyer**, a large observation wheel, visible in the center-right.<br>- The **ArtScience Museum** (the white sail-shaped building) between the Marina Bay Sands and the Flyer.<br>- The **Marina Bay waterfront** and the calm waters of Marina Bay reflecting the city lights.<br><br>These landmarks are all located in Singapore’s central business district along Marina Bay, one of the most famous and vibrant areas of the city. | 
| ![ChineseWall_S](https://github.com/user-attachments/assets/3ba4b57a-4b22-4050-b995-9599c6d239de) |**User**: \<image\>Tell me something about the history of this place.<br>**Answer**: This is the **Great Wall**, specifically the section near **Jiayuguan Pass** in Gansu Province, China — though the image appears to show a different segment (possibly the **Jinshanling or Mutianyu sections**) given the architecture and surrounding landscape.<br><br>However, since you’re asking about “this place” and the image shows a long, winding wall with watchtowers and terraced staircases nestled into green hills under a blue sky — it’s likely depicting one of China's most famous stretches: **Jinshanling Great Wall** or perhaps even **Mutianyu**, both near Beijing.<br><br>But to be precise, let me give you the general historical context:<br><br>🏯 The Great Wall – Historical Overview<br>The Great Wall is not a single structure but a series of walls and fortifications built over centuries by various Chinese dynasties — primarily the Qin (221–206 BCE), Han (206 BCE–220 CE), Ming (1368–1644), and later Qing.<br><br>🏛️ Key Historical Points:<br>- **Qin Dynasty (221–206 BCE)**: Emperor Qin Shi Huang ordered the unification of existing walls into a single, continuous barrier to defend against northern nomadic tribes — notably the Xiongnu. This was the first major construction phase.<br>- **Han Dynasty**: Extended the wall further north and added more watchtowers and fortresses.<br>- **Ming Dynasty (1368–1644)**: The most famous and well-preserved section of the Great Wall today was built during this time. The Ming emperors commissioned massive construction, using advanced engineering techniques to integrate the wall into mountainous terrain — as seen in your image. They also added **watchtowers**, **fortified gates**, and **stone staircases** for troop movement.<br><br>- **Purpose**: To protect China’s northern borders from invasions, control trade (especially along the Silk Road), and monitor nomadic tribes.<br><br>🏞️ What You’re Seeing<br>The image likely shows a section of the **Ming Great Wall**, possibly:<br>- **Jinshanling** — known for its scenic beauty and well-preserved architecture.<br>- Or **Mutianyu** — famous for its “crescent moon” shape and accessibility.<br><br>Both sections feature:<br>- Stone watchtowers with traditional Chinese roofs<br>- Terraced staircases leading to the top of hills<br>- Walls built into natural ridges, blending seamlessly with the landscape<br><br>🏰 Cultural Significance<br>The Great Wall is not just a military structure — it’s a symbol of China’s resilience and engineering prowess. It has been designated as a **UNESCO World Heritage Site** since 1987.<br>It also represents the cultural fusion between Chinese civilization and nomadic cultures, serving as both defense and diplomacy tool.<br><br>📜 Fun Fact<br>The Great Wall is so long — over **21,000 kilometers (13,000 miles)** — that it’s longer than any other man-made structure in history. It was built by millions of laborers, many of whom were conscripted or forced to work under harsh conditions.<br><br>If you’re referring to a specific section not shown here, feel free to describe more details (like the architecture, location, or surrounding landscape), and I can give you even more precise information!<br><br>But for now — this is likely part of **the Ming Great Wall**, one of China’s most iconic historical landmarks.|
| ![Pizza_S](https://github.com/user-attachments/assets/26fd5da3-f491-4b48-978a-14eae0819247) |**User**: <image>What are the ingredients I need to make this?<br>**Answer**: To make a pizza like the one in the image (which looks like a classic **Margherita pizza**), you’ll need these core ingredients:<br>🍀 **Base Ingredients:**<br>1. **Dough**<br>- All-purpose flour (or 00 flour for best results)<br>- Warm water<br>- Yeast (active dry or instant yeast)<br>- Salt<br>- Olive oil<br>2. **Tomato Sauce**<br>- Fresh tomatoes (preferably San Marzano) or canned crushed tomatoes<br>- Garlic (optional, but common)<br>- Onion (optional)<br>- Olive oil<br>- Salt<br>- Black pepper<br>- Sugar (a pinch to balance acidity)<br>🧀 **Toppings:**<br>3. **Fresh Mozzarella Cheese**<br>- High-quality fresh mozzarella (preferably buffalo-style or “caciocavallo” for authentic taste)<br>- Or shredded mozzarella if you prefer<br>4. **Fresh Basil Leaves**<br>- 1–2 cups of fresh basil, torn or chopped<br>- *Optional: add a drizzle of olive oil to the basil before placing on pizza*<br>5. **Salt & Pepper (to taste)**<br>🍳 Optional Additions:<br>- A pinch of red pepper flakes for heat<br>- Fresh oregano or thyme (optional)<br>- Grated Parmesan cheese (optional, but adds depth)<br>💡 Pro Tips:<br>- Use a pizza stone or baking steel to get that crispy crust.<br>- Preheat oven to 450–500°F (230–260°C) for best results.<br>- Let the dough rest for at least 1 hour before baking.<br><br>This is a **simple, classic Margherita** pizza — just tomato sauce, mozzarella, and basil. It’s meant to be light, fresh, and delicious!<br><br>Let me know if you want a recipe or step-by-step instructions! 🍕<br><br>**User**: What is the first step?<br>**Answer**:The **first step** in making this classic Margherita pizza (like the one pictured) is:<br>✅ **Prepare the Dough**<br>You’ll need to make your own pizza dough — it’s the foundation of a great pizza.<br>Here’s how to start:<br>1. **Combine dry ingredients**: In a large bowl, mix together:<br>- 3 cups (250g) all-purpose flour or 00 flour<br>- 1 teaspoon salt<br>2. **Activate yeast** (if using active dry yeast):<br>- In a small bowl, combine ¼ cup warm water (~105°F/40°C) with 1 packet (or 2¼ teaspoons) of active dry yeast.<br>- Let sit for 5–10 minutes until foamy — this means the yeast is alive and ready.<br>3. **Mix dough**:<br>- Pour the yeast mixture into the flour bowl, add ½ cup warm water and 2 tablespoons olive oil.<br>- Stir with a wooden spoon or spatula until a shaggy dough forms.<br>- Turn out onto a floured surface and knead for 8–10 minutes until smooth and elastic.<br>4. **First rise**:<br>- Place the dough in an oiled bowl, cover with plastic wrap or a damp towel.<br>- Let it rest at room temperature for 1 to 2 hours (or overnight in the fridge for better flavor).<br><br>*Once your dough is risen and ready, you can proceed to make the sauce and assemble the pizza.*<br>**Why this matters**: A good crust starts with quality dough. Skipping or rushing this step will affect texture and taste.<br><br>Let me know if you want a full recipe — I’ll walk you through every step! 🍕|
| ![Titanic_S](https://github.com/user-attachments/assets/bf6da76c-c9a4-41c6-bea1-2b4411dd4013) |**User**: <image>Which movie is this?<br>**Answer**:This image is from the 2012 film **“Titanic”**, directed by James Cameron.<br><br>The scene shows Leonardo DiCaprio as Jack Dawson and Kate Winslet as Rose DeWitt Bukater, standing on the deck of the Titanic during a romantic moment at sunset. This iconic scene — with the couple embracing and looking out over the ocean — is one of the most memorable in cinema history.<br><br>Note: While this is often associated with the 1997 version (also directed by James Cameron), the image you’ve shown appears to be from the **2012 re-release** or a promotional still, which was actually a different film entirely. However, it’s important to clarify that **the original “Titanic” (1997) is the most famous and widely recognized version**, and this scene is universally associated with it.<br><br>So, while technically there are two films titled *Titanic* — one from 1997 and another in 2012 — the image you’ve shared is **from the 1997 film**. The 2012 release was actually a different movie entirely (a reimagining or sequel, though it’s not officially titled “Titanic”).<br><br>✅ Final Answer: **“Titanic” (1997)** — directed by James Cameron.<br><br>*(Note: There is also a 2012 film called "Titanic" which was actually a different movie — but the image you’ve shared is from the original 1997 version.)*|

## LLM example.
Notice the nice Markdown makeup of the answer!<br><br>
**User**: Can you make me a C++ call that embeds the Pythagorean theorem?<br><br>
**Answer**:Actually, **there is no “Pythagorean theorem” in C++** — it’s a mathematical formula, not a programming construct. But you *can* write a C++ function or program that **implements the Pythagorean theorem**, which states:

> In a right-angled triangle, the square of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the other two sides.

### 📐 Formula:
```
c² = a² + b²
→ c = √(a² + b²)
```

---

## ✅ Here’s a simple C++ function that uses the Pythagorean theorem:

```cpp
#include <iostream>
#include <cmath> // for sqrt()

double pythagoreanTheorem(double a, double b) {
    return std::sqrt(a * a + b * b);
}

int main() {
    double sideA = 3.0;
    double sideB = 4.0;

    double hypotenuse = pythagoreanTheorem(sideA, sideB);

    std::cout << "For sides " << sideA << " and " << sideB
              << ", the hypotenuse is: " << hypotenuse << std::endl;

    return 0;
}
```

---

## 🧪 Output:
```
For sides 3 and 4, the hypotenuse is: 5
```

---

### 💡 Notes:

- This function computes the **hypotenuse** given two legs of a right triangle.
- You can also write it to compute any side if you know the other two — e.g., `a = sqrt(c² - b²)` etc.
- The `std::sqrt()` is from `<cmath>` — standard math library.

---

## 🧠 Bonus: Version that checks for right triangle

```cpp
#include <iostream>
#include <cmath>

bool isRightTriangle(double a, double b, double c) {
    // Sort the sides to ensure we compare largest with sum of squares
    double sides[3] = {a, b, c};
    std::sort(sides, sides + 3);
    return std::abs(sides[0] * sides[0] + sides[1] * sides[1] - sides[2] * sides[2]) < 1e-9;
}

int main() {
    double a = 3.0, b = 4.0, c = 5.0;

    if (isRightTriangle(a, b, c)) {
        std::cout << "This is a right triangle!" << std::endl;
    } else {
        std::cout << "Not a right triangle." << std::endl;
    }

    return 0;
}
```

---

Let me know if you want to embed it in a class, use vectors, or even visualize it with graphics! 🎨📐 

------------

## **[Rock5GPT](https://rock5gpt.qengineering.eu)**
To get a taste, try our professional Qwen3 AI-chatbot running on a Rock 5C: https://rock5gpt.qengineering.eu
<img width="815" height="1151" alt="Rock5GPT" src="https://github.com/user-attachments/assets/3ce5ad31-bc2b-4513-8ac9-42be793a86db" /><br>

------------

[![paypal](https://qengineering.eu/images/TipJarSmall4.png)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=CPZTM5BB3FCYL) 




