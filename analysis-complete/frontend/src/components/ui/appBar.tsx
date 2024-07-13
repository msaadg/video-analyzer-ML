export const AppBar = () => {
  return (
    <div className="w-full flex justify-between border-b-2 border-slate-400 shadow-md">
      <div className="bg-sky-600 my-4 ml-8 mr-8 px-10 py-2 rounded cursor-pointer text-white shadow-md">Logo</div>
      <div className="flex justify-between m-4 py-2">
        <div className="px-4 hover:text-sky-600 cursor-pointer">Acceul</div>
        <div className="px-4 hover:text-sky-600 cursor-pointer">À propos de nous</div>
        <div className="px-4 hover:text-sky-600 cursor-pointer">Connexion</div>
      </div>
    </div>
  )
}