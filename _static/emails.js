/// set/get helpers based on https://www.w3schools.com/js/js_cookies.asp
function setCookie(cname, cvalue, exdays) {
  const d = new Date();
  d.setTime(d.getTime() + (exdays * 24 * 60 * 60 * 1000));
  document.cookie = cname + "=" + cvalue + ";expires=" + d.toUTCString() + ";SameSite=Strict;path=/";
}

function getCookie(cname) {
  let name = cname + "=";
  let ca = document.cookie.split(';');
  for (let i = 0; i < ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) === ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(name) === 0) {
      return c.substring(name.length, c.length);
    }
  }
  return "";
}

async function emailButtonClick() {
  let emailInput = document.getElementById("email-input");
  let emailValue = emailInput.value;
  let res = await fetch("https://premai.pythonanywhere.com/email?a=" + emailValue);
  const ok = 200 <= res.status && res.status < 299;
  const server_err = 500 <= res.status && res.status < 599;
  if (ok || server_err) {
    let modal = document.getElementById('email-modal');
    modal.style.display = 'none'
    emailInput.value = "";
  } else {
    let emailError = document.getElementsByClassName('email-error')[0];
    let msg = await res.json();
    emailError.innerHTML = "Error " + res.status + ": " + msg.status;
  }
  if (ok) {
    setCookie("address", emailValue, 365); // might fail if cookies disabled
  }
}

document.addEventListener('DOMContentLoaded', function() {
  let modal = document.getElementById('email-modal');
  let email = getCookie("address");
  let emailInput = document.getElementById("email-input");
  if (email === "" || email == null) {
    modal.style.display = 'block';
    emailInput.value = "";
  }
  emailInput.focus()
  // When user click Enter, click the submit button
  emailInput.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      document.getElementById("email-submit").click();
    }
  });
});
