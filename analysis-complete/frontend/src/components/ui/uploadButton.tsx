import React, { useState } from 'react';
import axios, { AxiosProgressEvent } from 'axios';

interface UploadButtonProps {
  setUploadProgress: (progress: number) => void;
  setVideoUploaded: (uploaded: boolean) => void;
  setUploadedFilePath: (path: string) => void;
}

export const UploadButton: React.FC<UploadButtonProps> = ({ setUploadProgress, setVideoUploaded, setUploadedFilePath }) => {
  const [uploading, setUploading] = useState(false);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    axios.post('http://localhost:3000/upload', formData, {
      onUploadProgress: (progressEvent: AxiosProgressEvent) => {
        if (progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        }
      }
    })
    .then(response => {
      setUploading(false);
      setVideoUploaded(true);
      setUploadProgress(100); // Ensure that the progress is set to 100 on successful upload
      setUploadedFilePath(response.data.file_path); // Store the file path from the response
      console.log('File uploaded successfully', response);
    })
    .catch(error => {
      setUploading(false);
      console.error('Error uploading file', error);
    });
  };

  return (
    <label htmlFor="uploadFile1"
      className={`flex bg-sky-600 hover:bg-sky-700 text-white px-5 py-3 m-4 rounded-lg w-max cursor-pointer mx-auto shadow-lg ${uploading ? 'opacity-50 cursor-not-allowed' : ''}`}>
      <svg xmlns="http://www.w3.org/2000/svg" className="w-6 mr-2 fill-white inline" viewBox="0 0 32 32">
        <path
          d="M23.75 11.044a7.99 7.99 0 0 0-15.5-.009A8 8 0 0 0 9 27h3a1 1 0 0 0 0-2H9a6 6 0 0 1-.035-12 1.038 1.038 0 0 0 1.1-.854 5.991 5.991 0 0 1 11.862 0A1.08 1.08 0 0 0 23 13a6 6 0 0 1 0 12h-3a1 1 0 0 0 0 2h3a8 8 0 0 0 .75-15.956z"/>
        <path
          d="M20.293 19.707a1 1 0 0 0 1.414-1.414l-5-5a1 1 0 0 0-1.414 0l-5 5a1 1 0 0 0 1.414 1.414L15 16.414V29a1 1 0 0 0 2 0V16.414z"/>
      </svg>
      Choisir
      <input type="file" id='uploadFile1' className="hidden" onChange={handleFileUpload} disabled={uploading} />
    </label>
  );
};
