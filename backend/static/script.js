

let currentChatId = null;
let isLoadingChats = false;
async function updateUserUI() {
  const res = await fetch('/me');
  const data = await res.json();
  const userDiv = document.getElementById('userStatus');
  userDiv.innerHTML = ''; // clear existing content

  if (data.logged_in) {
    const initial = data.email.charAt(0).toUpperCase();

    // Create the circle
    const circle = document.createElement("div");
    circle.className = "user-initial";
    circle.innerText = initial;
    circle.style.position = "relative"; // for dropdown positioning
    userDiv.appendChild(circle);

    // Create dropdown container (hidden by default)
    const dropdown = document.createElement("div");
    dropdown.className = "dropdown-menu";
    dropdown.style.display = "none";
    dropdown.style.position = "absolute";
    dropdown.style.top = "110%";
    dropdown.style.right = "0";
    dropdown.style.background = "#fff";
    dropdown.style.border = "1px solid #ccc";
    dropdown.style.borderRadius = "5px";
    dropdown.style.boxShadow = "0 2px 8px rgba(0,0,0,0.2)";
    dropdown.style.minWidth = "100px";
    dropdown.style.zIndex = "100";
    
    // Add logout option
    const logoutBtn = document.createElement("div");
    logoutBtn.textContent = "Logout";
    logoutBtn.style.padding = "10px";
    logoutBtn.style.cursor = "pointer";
    logoutBtn.style.color="black";
    logoutBtn.addEventListener("click", async () => {
      await fetch('/logout');
      window.location.reload();
    });
    logoutBtn.addEventListener("mouseover", () => logoutBtn.style.background = "#eee");
    logoutBtn.addEventListener("mouseout", () => logoutBtn.style.background = "#fff");

    dropdown.appendChild(logoutBtn);
    circle.appendChild(dropdown);

    // Toggle dropdown on circle click
    circle.addEventListener("click", (e) => {
      e.stopPropagation(); // prevent event bubbling
      dropdown.style.display = dropdown.style.display === "none" ? "block" : "none";
    });

    // Close dropdown if clicked outside
    document.addEventListener("click", () => {
      dropdown.style.display = "none";
    });

  } else {
    // show login button
    const btn = document.createElement('button');
    btn.className = 'login-btn';
    btn.textContent = 'Login';
    btn.onclick = () => {
      window.location.href = '/login';
    };
    userDiv.appendChild(btn);
  }

  document.getElementById("sendBtn").addEventListener("click", sendMessage);
  document.getElementById("userInput").addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
  });
}


function speakText(text) {
    const speech = new SpeechSynthesisUtterance(text);
    speech.lang = 'en-US';
    speech.rate = 1; // speed
    speech.pitch = 1; // tone
    window.speechSynthesis.speak(speech);
}

// Load all chats on page load
async function loadChats(autoSelect = true) {
  isLoadingChats = true;
  const res = await fetch("/chats");
  const data = await res.json();
  const list = document.getElementById("chatList");
  list.innerHTML = "";

  // Create list items
  data.chats.forEach((chat, idx) => {
    const li = document.createElement("li");
    li.textContent = chat.title;
    li.dataset.chatId = chat.id;
    li.onclick = () => {
      loadChat(chat.id);
    };
    list.appendChild(li);
  });

  // If there are chats and autoSelect is true, pick the first (most recent)
  if (autoSelect && data.chats && data.chats.length > 0) {
    const firstId = data.chats[0].id;
    // small timeout to let the UI render
    setTimeout(() => loadChat(firstId), 50);
  } else if (autoSelect && (!data.chats || data.chats.length === 0)) {
    // No chats yet -> create one automatically
    await newChat();
  }
  isLoadingChats = false;
}

// Create new chat
async function newChat() {
  const res = await fetch("/new_chat", { method: "POST" });
  const data = await res.json();
  currentChatId = data.chat_id;
  document.getElementById("messages").innerHTML = "";
  await loadChats(false); // refresh list but don't auto-select (we already set currentChatId)
  // visually highlight newly created chat (optional)
  highlightSelectedChat();
}

// Load chat messages
async function loadChat(id) {
  // ⭐ Guest ko messages load nahi karna
  const loggedIn = await checkLoginStatus();
  if (!loggedIn) {
    document.getElementById("messages").innerHTML = "";
    return;
  }

  if (!id) return;

  currentChatId = id;
  highlightSelectedChat();

  const res = await fetch(`/messages/${id}`);
  const data = await res.json();
  const msgDiv = document.getElementById("messages");
  msgDiv.innerHTML = "";
  
  data.messages.forEach(m => appendMessage(m.sender, m.message));
}


// Highlight selected chat in the sidebar
function highlightSelectedChat() {
  const lis = document.querySelectorAll("#chatList li");
  lis.forEach(li => {
    if (li.dataset.chatId == currentChatId) {
      li.style.background = "#e6f0ff";
    } else {
      li.style.background = "";
    }
  });
}

// Append message to chat UI
function appendMessage(sender, text) {
  const div = document.createElement("div");
  div.className = sender === "user" ? "user" : "ai";
  div.innerHTML = `<strong>${sender === "user" ? "You" : "AI"}:</strong> ${text}`;
  document.getElementById("messages").appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
}

async function checkLoginStatus() {
  const res = await fetch('/me');
  const data = await res.json();
  return data.logged_in;
}


// Ensure there's a chat id; if not create one automatically
async function ensureChatExists() {
  if (currentChatId) return currentChatId;
  // If we're already loading chats, wait a bit
  if (isLoadingChats) {
    await new Promise(r => setTimeout(r, 100));
    if (currentChatId) return currentChatId;
  }
  // Create a new chat
  const res = await fetch("/new_chat", { method: "POST" });
  const data = await res.json();
  currentChatId = data.chat_id;
  await loadChats(false);
  return currentChatId;
}

// Send user message
async function sendMessage() {
  const input = document.getElementById("userInput");
  const text = input.value.trim();
  if (!text) return;

  // ensure a chat exists
  await ensureChatExists();

  // append locally right away for fast UI feedback
  appendMessage("user", text);
  input.value = "";

  const res = await fetch(`/chat/${currentChatId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text }),
  });
  
  const data = await res.json();
  appendMessage("ai", data.reply);
  speakText(data.reply);
  // refresh list so sidebar titles stay updated
}

// Bind events
document.getElementById("sendBtn").onclick = sendMessage;
document.getElementById("newChat").onclick = newChat;
document.getElementById("userInput").addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});
const micBtn = document.getElementById("mic-btn");
const userInput = document.getElementById("userInput");

  micBtn.addEventListener("click", () => {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = "en-IN"; // or "en-US"
    recognition.start();

    recognition.onresult = (event) => {
      const speechText = event.results[0][0].transcript;
      userInput.value = speechText;
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
    };
  });

// Initial load
// Initial load
// Initial load sequence


document.addEventListener("DOMContentLoaded", async () => {
  await checkLoginStatus(); // check and render correct UI (login/logout)
  await loadChats();        // load chat history after login
  await updateUserUI();
  // Bind buttons here
  
});
