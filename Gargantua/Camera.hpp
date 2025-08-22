#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>

class Camera
{
public:
	Camera()
	{
		position = glm::vec3(0.0f, 0.0f, 3.0f);
		worldUp = glm::vec3(0.0f, 1.0f, 0.0f); // Y-up
		yaw = -90.0f; // To look along -Z axis initially
		pitch = 0.0f;
		front = glm::vec3(0.0f, 0.0f, -1.0f); // Will be properly set by updateCameraVectors

		moveSpeed = 5.0f;   // Sensible default
		turnSpeed = 0.1f;   // Sensible default

		fov = 45.0f;
		// A common default, but will be overridden if parameterized constructor is used
		aspectRatio = 16.0f / 9.0f;
		near = 0.1f;
		far = 100.0f;
		updateCameraVectors();
	}
	Camera(glm::vec3 startPosition, glm::vec3 worldUpVector, float startYaw, float startPitch, float startMoveSpeed, float startTurnSpeed, float fov, float aspectRatio, float near, float far)
	{
		position = startPosition;
		worldUp = worldUpVector;
		yaw = startYaw;
		pitch = startPitch;
		front = glm::vec3(0.0f, 0.0f, -1.0f);

		moveSpeed = startMoveSpeed;
		turnSpeed = startTurnSpeed;

		this->fov = fov;
		this->aspectRatio = aspectRatio;
		this->near = near;
		this->far = far;
		updateCameraVectors();
	}
	~Camera()
	{

	}

	void processKeyboard(bool* keys, float deltaTime)
	{
		float velocity = moveSpeed * deltaTime;
		if (keys[GLFW_KEY_W])
		{
			position += front * velocity;
		}
		if (keys[GLFW_KEY_S])
		{
			position -= front * velocity;
		}
		if (keys[GLFW_KEY_D])
		{
			position += right * velocity;
		}
		if (keys[GLFW_KEY_A])
		{
			position -= right * velocity;
		}
		if (keys[GLFW_KEY_LEFT_SHIFT])
		{
			position -= worldUp * velocity;
		}
		if (keys[GLFW_KEY_SPACE])
		{
			position += worldUp * velocity;
		}
	}

	void processMouseMovement(float xoffset, float yoffset, bool constrainPitch)
	{
		xoffset *= turnSpeed;
		yoffset *= turnSpeed;

		yaw += xoffset;
		pitch += yoffset;

		if (constrainPitch)
		{
			if (pitch > 89.0f)
			{
				pitch = 89.0f;
			}

			if (pitch < -89.0)
			{
				pitch = -89.0f;
			}
		}

		updateCameraVectors();
	}

	glm::vec3 getCameraPosition() const
	{
		return position;
	}
	glm::vec3 getCameraDirection() const
	{
		return glm::normalize(front);
	}
	glm::mat4 calculateViewMatrix() const
	{
		return glm::lookAt(position, position + front, up);
	}
	glm::mat4 getProjectionMatrix() const
	{
		return glm::perspective(glm::radians(fov), aspectRatio, near, far);
	}

	void setAspectRatio(float aspectRatio)
	{
		this->aspectRatio = aspectRatio;
	}

private:
	glm::vec3 position;
	glm::vec3 front;
	glm::vec3 up;
	glm::vec3 right;
	glm::vec3 worldUp;

	float yaw;
	float pitch;

	float moveSpeed;
	float turnSpeed;

	float fov;
	float aspectRatio;
	float near;
	float far;

	void updateCameraVectors()
	{
		// Calculate the new Front vector based on Yaw and Pitch (Y-UP system)
		glm::vec3 tempFront;
		tempFront.x = glm::cos(glm::radians(yaw)) * glm::cos(glm::radians(pitch));
		tempFront.y = glm::sin(glm::radians(pitch));
		tempFront.z = glm::sin(glm::radians(yaw)) * glm::cos(glm::radians(pitch));
		front = glm::normalize(tempFront);

		// 'worldUp' is the up direction of the world (e.g., glm::vec3(0.0f, 1.0f, 0.0f))
		right = glm::normalize(glm::cross(front, worldUp));
		up = glm::normalize(glm::cross(right, front)); // Camera's local up
	}
};




//#pragma once
//
//#define GLM_FORCE_RADIANS
//#define GLM_FORCE_DEPTH_ZERO_TO_ONE
//
//#include <glm/glm.hpp>
//#include <glm/gtc/quaternion.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//
//class Camera
//{
//private:
//	float fov;
//	float znear, zfar;
//
//	void updateViewMatrix()
//	{
//		glm::mat4 currentMatrix = matrices.view;
//
//		glm::mat4 rotM = glm::mat4(1.0f);
//		glm::mat4 transM;
//
//		rotM = glm::rotate(rotM, glm::radians(rotation.x * (flipY ? -1.0f : 1.0f)), glm::vec3(1.0f, 0.0f, 0.0f));
//		rotM = glm::rotate(rotM, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
//		rotM = glm::rotate(rotM, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
//
//		glm::vec3 translation = position;
//		if (flipY)
//		{
//			translation.y *= -1.0f;
//		}
//		transM = glm::translate(glm::mat4(1.0f), translation);
//
//		if (type == CameraType::firstperson)
//		{
//			matrices.view = rotM * transM;
//		}
//		else
//		{
//			matrices.view = transM * rotM;
//		}
//
//		viewPos = glm::vec4(position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);
//
//		if (matrices.view != currentMatrix)
//		{
//			updated = true;
//		}
//	}
//
//public:
//	enum CameraType { lookat, firstperson };
//	CameraType type = CameraType::lookat;
//
//	glm::vec3 rotation = glm::vec3();
//	glm::vec3 position = glm::vec3();
//	glm::vec4 viewPos = glm::vec4();
//
//	float rotationSpeed = 1.0f;
//	float movementSpeed = 1.0f;
//
//	bool updated = true;
//	bool flipY = false;
//
//	struct {
//		glm::mat4 perspective;
//		glm::mat4 view;
//	} matrices;
//
//	struct {
//		bool left = false;
//		bool right = false;
//		bool up = false;
//		bool down = false;
//	} keys;
//
//	bool moving() const
//	{
//		return keys.left || keys.right || keys.up || keys.down;
//	}
//
//	float getNearClip() const
//	{
//		return znear;
//	}
//
//	float getFarClip() const
//	{
//		return zfar;
//	}
//
//	float getFov() const
//	{
//		return fov;
//	}
//
//	void setPerspective(float fov, float aspect, float znear, float zfar)
//	{
//		glm::mat4 currentMatrix = matrices.perspective;
//		this->fov = fov;
//		this->znear = znear;
//		this->zfar = zfar;
//		matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
//		if (flipY)
//		{
//			matrices.perspective[1][1] *= -1.0f;
//		}
//		if (matrices.view != currentMatrix)
//		{
//			updated = true;
//		}
//	}
//
//	void updateAspectRatio(float aspect)
//	{
//		glm::mat4 currentMatrix = matrices.perspective;
//		matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
//		if (flipY)
//		{
//			matrices.perspective[1][1] *= -1.0f;
//		}
//		if (matrices.view != currentMatrix)
//		{
//			updated = true;
//		}
//	}
//
//	
//
//	void setPosition(glm::vec3 position)
//	{
//		this->position = position;
//		updateViewMatrix();
//	}
//
//	void setRotation(glm::vec3 rotation)
//	{
//		this->rotation = rotation;
//		updateViewMatrix();
//	}
//
//	void rotate(glm::vec3 delta)
//	{
//		this->rotation += delta;
//		updateViewMatrix();
//	}
//
//	void setTranslation(glm::vec3 translation)
//	{
//		this->position = translation;
//		updateViewMatrix();
//	}
//
//	void translate(glm::vec3 delta)
//	{
//		this->position += delta;
//		updateViewMatrix();
//	}
//
//	void setRotationSpeed(float rotationSpeed)
//	{
//		this->rotationSpeed = rotationSpeed;
//	}
//
//	void setMovementSpeed(float movementSpeed)
//	{
//		this->movementSpeed = movementSpeed;
//	}
//
//	void update(float deltaTime)
//	{
//		updated = false;
//		if (type == CameraType::firstperson)
//		{
//			if (moving())
//			{
//				glm::vec3 camFront;
//				camFront.x = -cos(glm::radians(rotation.x)) * sin(glm::radians(rotation.y));
//				camFront.y = sin(glm::radians(rotation.x));
//				camFront.z = cos(glm::radians(rotation.x)) * cos(glm::radians(rotation.y));
//				camFront = glm::normalize(camFront);
//
//				float moveSpeed = deltaTime * movementSpeed;
//
//				if (keys.up) position += camFront * moveSpeed;
//				if (keys.down) position -= camFront * moveSpeed;
//				if (keys.left) position -= glm::normalize(glm::cross(camFront, glm::vec3(0.0, 1.0f, 0.0f))) * moveSpeed;
//				if (keys.right) position += glm::normalize(glm::cross(camFront, glm::vec3(0.0, 1.0f, 0.0f))) * moveSpeed;
//			}
//		}
//		updateViewMatrix();
//	}
//};