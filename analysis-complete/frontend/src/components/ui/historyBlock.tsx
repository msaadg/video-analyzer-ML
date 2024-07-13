import { SmallVideo } from "./smallVideo"

export const HistoryBlock = () => {
  return (
    <div className="">
      <div className="flex justify-center text-md font-medium text-gray-900">Historique</div>
      <div className="mt-4 flex justify-between">
        <SmallVideo />
        <SmallVideo />
        <SmallVideo />
      </div>
    </div>
  )
}