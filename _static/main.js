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

document.addEventListener('DOMContentLoaded', function() {
  // Show the email modal if the user has not entered their email
  let modal = document.getElementById('email-modal');
  let emailCookie = getCookie("email");
  let emailInput = document.getElementById("email-input");
  if (emailCookie === false || emailCookie === "" || emailCookie === null) {
    modal.style.display = 'flex';
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

  // Handle form submission
  const form = document.getElementById('email-form');
  form.addEventListener('submit', async function(event) {
    event.preventDefault();
    const formData = new FormData(form);
    const email = formData.get('email');
    if (!email) return;
    try {
      const response = await fetch('https://state-of-open-source-ai.vercel.app/api/add-member', {
        method: 'POST',
        body: JSON.stringify({ email }),
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const responseData = await response.json();

      if (response.ok) {
        let modal = document.getElementById('email-modal');
        modal.style.display = 'none';
        setCookie("email", emailInput.value, 365); // might fail if cookies disabled
      } else {
        document.querySelector('.email-error').textContent = responseData.error || 'An unexpected error occurred. Please enter a valid email.';
      }
    } catch (error) {
      console.error('Error:', error);
      document.querySelector('.email-error').textContent = 'An unexpected error occurred. Please try again.';
    }
  });
});
