variable "name" {
  description = "Repository name."
  type        = string
}

variable "keep_images" {
  description = "How many images to keep before older ones expire."
  type        = number
  default     = 10
}
