#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <string>
#include <functional>
#include <stdexcept>

class Window
{
public:
	using AppFramebufferResizeCallback = std::function<void(int width, int height)>;
	using FramebufferResizeCallbackUserFn = std::function<void>(GLFWwindow*, int, int);

	Window(int width, int height, const std::string& title);
	~Window();

	Window(const Window&) = delete;
	Window& operator=(const Window&) = delete;
	Window(Window&&) = delete;
	Window& operator=(Window&&) = delete;

	bool shouldClose() const;
	void pollEvents() const;

	void getFramebufferSize(int& width, int& height) const;
	GLFWwindow* getGlfwWindow() const { return glfwWindow; }
	bool* getKeys() { return keys; }
	GLfloat getXChange();
	GLfloat getYChange();

	void getCursorPos(double& x, double& y);

	void waitForRestoredSize();

	void setFramebufferSizeCallback(GLFWframebuffersizefun callback);
	void setUserPointer(void* pointer);

	void setAppFramebufferResizeCallback(AppFramebufferResizeCallback callback);

	void endFrame();
	bool isKeyPressed(int key) const;
	bool isKeyTriggered(int key) const;

	bool isMouseButtonPressed(int button) const;
	bool isMouseButtonTriggered(int button)const;

private:
	GLFWwindow* glfwWindow;
	std::string windowTitle;

	bool keys[1024];
	bool lastKeys[1024];
	bool mouseButtons[GLFW_MOUSE_BUTTON_LAST + 1];
	bool lastMouseButtons[GLFW_MOUSE_BUTTON_LAST + 1];

	GLfloat lastX;
	GLfloat lastY;
	GLfloat xChange;
	GLfloat yChange;
	bool mouseFirstMoved;

	AppFramebufferResizeCallback appResizeCallbackInstance;

	void createCallbacks();
	static void staticKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void staticMouseCallback(GLFWwindow* window, double xPos, double yPos);
	static void staticFramebufferResizeCallback(GLFWwindow* glfwWnd, int width, int height);

	static void staticMouseButtonCallback(GLFWwindow* window, int button, int action, int mods);


};