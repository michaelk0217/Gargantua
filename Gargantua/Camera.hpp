#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera
{
private:
	float fov;
	float znear, zfar;

	void updateViewMatrix()
	{
		glm::mat4 currentMatrix = matrices.view;

		glm::mat4 rotM = glm::mat4(1.0f);
		glm::mat4 transM;

		rotM = glm::rotate(rotM, glm::radians(rotation.x * (flipY ? -1.0f : 1.0f)), glm::vec3(1.0f, 0.0f, 0.0f));
		rotM = glm::rotate(rotM, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		rotM = glm::rotate(rotM, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		glm::vec3 translation = position;
		if (flipY)
		{
			translation.y *= -1.0f;
		}
		transM = glm::translate(glm::mat4(1.0f), translation);

		if (type == CameraType::firstperson)
		{
			matrices.view = rotM * transM;
		}
		else
		{
			matrices.view = transM * rotM;
		}

		viewPos = glm::vec4(position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

		if (matrices.view != currentMatrix)
		{
			updated = true;
		}
	}

public:
	enum CameraType { lookat, firstperson };
	CameraType type = CameraType::lookat;

	glm::vec3 rotation = glm::vec3();
	glm::vec3 position = glm::vec3();
	glm::vec4 viewPos = glm::vec4();

	float rotationSpeed = 1.0f;
	float movementSpeed = 1.0f;

	bool updated = true;
	bool flipY = false;

	struct {
		glm::mat4 perspective;
		glm::mat4 view;
	} matrices;

	struct {
		bool left = false;
		bool right = false;
		bool up = false;
		bool down = false;
	} keys;

	bool moving() const
	{
		return keys.left || keys.right || keys.up || keys.down;
	}

	float getNearClip() const
	{
		return znear;
	}

	float getFarClip() const
	{
		return zfar;
	}

	float getFov() const
	{
		return fov;
	}

	void setPerspective(float fov, float aspect, float znear, float zfar)
	{
		glm::mat4 currentMatrix = matrices.perspective;
		this->fov = fov;
		this->znear = znear;
		this->zfar = zfar;
		matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
		if (flipY)
		{
			matrices.perspective[1][1] *= -1.0f;
		}
		if (matrices.view != currentMatrix)
		{
			updated = true;
		}
	}

	void updateAspectRatio(float aspect)
	{
		glm::mat4 currentMatrix = matrices.perspective;
		matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
		if (flipY)
		{
			matrices.perspective[1][1] *= -1.0f;
		}
		if (matrices.view != currentMatrix)
		{
			updated = true;
		}
	}

	void setPosition(glm::vec3 position)
	{
		this->position = position;
		updateViewMatrix();
	}

	void setRotation(glm::vec3 rotation)
	{
		this->rotation = rotation;
		updateViewMatrix();
	}

	void rotate(glm::vec3 delta)
	{
		this->rotation += delta;
		updateViewMatrix();
	}

	void setTranslation(glm::vec3 translation)
	{
		this->position = translation;
		updateViewMatrix();
	}

	void translate(glm::vec3 delta)
	{
		this->position += delta;
		updateViewMatrix();
	}

	void setRotationSpeed(float rotationSpeed)
	{
		this->rotationSpeed = rotationSpeed;
	}

	void setMovementSpeed(float movementSpeed)
	{
		this->movementSpeed = movementSpeed;
	}

	void update(float deltaTime)
	{
		updated = false;
		if (type == CameraType::firstperson)
		{
			if (moving())
			{
				glm::vec3 camFront;
				camFront.x = -cos(glm::radians(rotation.x)) * sin(glm::radians(rotation.y));
				camFront.y = sin(glm::radians(rotation.x));
				camFront.z = cos(glm::radians(rotation.x)) * cos(glm::radians(rotation.y));
				camFront = glm::normalize(camFront);

				float moveSpeed = deltaTime * movementSpeed;

				if (keys.up) position += camFront * moveSpeed;
				if (keys.down) position -= camFront * moveSpeed;
				if (keys.left) position -= glm::normalize(glm::cross(camFront, glm::vec3(0.0, 1.0f, 0.0f))) * moveSpeed;
				if (keys.right) position += glm::normalize(glm::cross(camFront, glm::vec3(0.0, 1.0f, 0.0f))) * moveSpeed;
			}
		}
		updateViewMatrix();
	}
};