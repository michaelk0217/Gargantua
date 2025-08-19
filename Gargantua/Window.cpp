#include "Window.h"
#include <iostream>

Window::Window(int width, int height, const std::string& title) : windowTitle(title)
{
	if (!glfwInit())
	{
		throw std::runtime_error("Failed to initialize GLFW!");
	}

	for (int i = 0; i < 1024; ++i)
	{
		keys[i] = false;
		lastKeys[i] = false;
	}
	for (int i = 0; i <= GLFW_MOUSE_BUTTON_LAST; ++i)
	{
		mouseButtons[i] = false;
		lastMouseButtons[i] = false;
	}
	xChange = 0.0f;
	yChange = 0.0f;
	mouseFirstMoved = true;

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindow = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
	if (!glfwWindow)
	{
		glfwTerminate();
		throw std::runtime_error("Failed to create GLFW window!");
	}

	glfwSetWindowUserPointer(glfwWindow, this);
	//glfwSetInputMode(glfwWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	createCallbacks();

}

Window::~Window()
{
	if (glfwWindow)
	{
		glfwDestroyWindow(glfwWindow);
	}

	glfwTerminate();
}

bool Window::shouldClose() const
{
	if (!glfwWindow) return true;

	return glfwWindowShouldClose(glfwWindow);
}

void Window::pollEvents() const
{
	glfwPollEvents();
}



void Window::getFramebufferSize(int& width, int& height) const
{
	if (!glfwWindow)
	{
		width = 0;
		height = 0;
		return;
	}

	glfwGetFramebufferSize(glfwWindow, &width, &height);
}

GLfloat Window::getXChange()
{
	GLfloat theChange = xChange;
	xChange = 0.0f;
	return theChange;
}

GLfloat Window::getYChange()
{
	GLfloat theChange = yChange;
	yChange = 0.0f;
	return theChange;
}

void Window::getCursorPos(double& x, double& y)
{
	glfwGetCursorPos(glfwWindow, &x, &y);
}

void Window::setFramebufferSizeCallback(GLFWframebuffersizefun callback)
{
	if (!glfwWindow) return;
	glfwSetFramebufferSizeCallback(glfwWindow, callback);
}

void Window::setUserPointer(void* pointer)
{
	if (!glfwWindow) return;
	glfwSetWindowUserPointer(glfwWindow, pointer);
}

void Window::waitForRestoredSize()
{
	if (!glfwWindow) return;
	int width = 0, height = 0;
	getFramebufferSize(width, height);
	while (width == 0 || height == 0)
	{
		getFramebufferSize(width, height);
		glfwWaitEvents();
	}
}

void Window::createCallbacks()
{
	glfwSetKeyCallback(glfwWindow, Window::staticKeyCallback);
	glfwSetCursorPosCallback(glfwWindow, Window::staticMouseCallback);
	glfwSetFramebufferSizeCallback(glfwWindow, Window::staticFramebufferResizeCallback);
	glfwSetMouseButtonCallback(glfwWindow, Window::staticMouseButtonCallback);
}

void Window::staticKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	Window* thisWindow = static_cast<Window*>(glfwGetWindowUserPointer(window));
	if (!thisWindow) return;

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
	if (key >= 0 && key < 1024) {
		if (action == GLFW_PRESS) {
			thisWindow->keys[key] = true;
			//printf("Pressed: %d\n", key);
		}
		else if (action == GLFW_RELEASE) {
			thisWindow->keys[key] = false;
			//printf("Released: %d\n", key);
		}
	}
}

void Window::staticMouseCallback(GLFWwindow* window, double xPos, double yPos) {
	Window* thisWindow = static_cast<Window*>(glfwGetWindowUserPointer(window));
	if (!thisWindow) return;

	if (thisWindow->mouseFirstMoved) {
		thisWindow->lastX = static_cast<GLfloat>(xPos);
		thisWindow->lastY = static_cast<GLfloat>(yPos);
		thisWindow->mouseFirstMoved = false;
	}
	thisWindow->xChange = static_cast<GLfloat>(xPos) - thisWindow->lastX;
	thisWindow->yChange = thisWindow->lastY - static_cast<GLfloat>(yPos); // Typical Y-inversion
	thisWindow->lastX = static_cast<GLfloat>(xPos);
	thisWindow->lastY = static_cast<GLfloat>(yPos);
}

void Window::setAppFramebufferResizeCallback(AppFramebufferResizeCallback callback)
{
	appResizeCallbackInstance = callback;
}

void Window::endFrame()
{
	memcpy(lastKeys, keys, sizeof(keys));
	memcpy(lastMouseButtons, mouseButtons, sizeof(mouseButtons));
}

bool Window::isKeyPressed(int key) const
{
	return keys[key];
}

bool Window::isKeyTriggered(int key) const
{
	return keys[key] && !lastKeys[key];
}

bool Window::isMouseButtonPressed(int button) const
{
	return mouseButtons[button];
}

bool Window::isMouseButtonTriggered(int button) const
{
	return mouseButtons[button] && !lastMouseButtons[button];
}

void Window::staticFramebufferResizeCallback(GLFWwindow* glfwWnd, int width, int height) {
	Window* thisWindow = static_cast<Window*>(glfwGetWindowUserPointer(glfwWnd));
	if (thisWindow) {
		// You can add any internal logic Window needs on resize here
		// For example: glViewport(0, 0, width, height); if it were an OpenGL window

		// Now, call the application's registered callback, if any
		if (thisWindow->appResizeCallbackInstance) {
			thisWindow->appResizeCallbackInstance(width, height);
		}
	}
}

void Window::staticMouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	Window* thisWindow = static_cast<Window*>(glfwGetWindowUserPointer(window));
	if (!thisWindow) return;

	if (button >= 0 && button <= GLFW_MOUSE_BUTTON_LAST)
	{
		if (action == GLFW_PRESS)
		{
			thisWindow->mouseButtons[button] = true;
			//std::cout << "mouse button pressed" << std::endl;
		}
		else if (action == GLFW_RELEASE)
		{
			thisWindow->mouseButtons[button] = false;
			//std::cout << "mouse button released" << std::endl;
		}
	}
}